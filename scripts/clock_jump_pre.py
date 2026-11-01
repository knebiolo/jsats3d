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
import datetime as dt
import os
import sqlite3
import sys

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


def find_events(con, receiver, t0, t1):
    """Step 1: internal evidence rows for one receiver in [t0, t1).

    Jump candidates (triggers) are OffsetChanged (the file's own 'clock offset changed'
    code: first detection after an InternalOffset change within the file) and
    CounterRestart rows. OneSecondAdjustmentEvidence is dense (fires on most pings;
    median gap ~2 s, 2026-10-29 audit of the 2025 early-fish DB) so it corroborates a
    nearby candidate instead of triggering one.

    Returns (candidates, all_evidence): the sparse trigger rows (each with a
    one_sec_nearby corroboration flag) and every evidence row (for counting).
    """
    ev = pd.read_sql(
        """SELECT seconds, OffsetChanged, CounterRestart, OneSecondAdjustmentEvidence,
                  InternalOffsetValue, SourceFile
           FROM tblDetectionRaw
           WHERE Rec_ID = ? AND seconds >= ? AND seconds < ?
             AND (OffsetChanged = 1 OR CounterRestart = 1 OR OneSecondAdjustmentEvidence = 1)
           ORDER BY seconds""", con, params=[receiver, t0, t1])
    if ev.empty:
        return ev, ev
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


def _evidence_string(row):
    parts = [name for name, flag in (("OffsetChanged", row["OffsetChanged"]),
                                     ("CounterRestart", row["CounterRestart"])) if int(flag)]
    if int(row.get("one_sec_nearby", row["OneSecondAdjustmentEvidence"])):
        parts.append("OneSecondAdjustment")
    return ";".join(parts)


def quantify_jumps(con, receivers, master_rec, beacon_tag, period, t0, t1,
                   window_s=300.0, min_step=0.05, max_step=5.0):
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
        cands, all_ev = find_events(con, rec, t0, t1)
        n_events += int(len(all_ev))
        if cands.empty:
            continue
        local = arrivals[arrivals.Rec_ID == rec]
        cands = cands.reset_index(drop=True)
        for i, row in cands.iterrows():
            t_ev = float(row["seconds"])
            t_start = t_ev - window_s
            if i > 0:
                t_start = max(t_start, float(cands.loc[i - 1, "seconds"]))
            t_end = t_ev + window_s
            if i + 1 < len(cands):
                t_end = min(t_end, float(cands.loc[i + 1, "seconds"]))
            pre = local[(local.t >= t_start) & (local.t < t_ev)]
            post = local[(local.t > t_ev) & (local.t <= t_end)]
            base = {"Rec_ID": rec, "jump_time_s": t_ev, "jump_time_local": _local_ts(t_ev).strftime("%Y-%m-%d %H:%M:%S"),
                    "evidence": _evidence_string(row), "source_file": row["SourceFile"],
                    "n_pre": int(len(pre)), "n_post": int(len(post))}
            # A full +/-window_s side holds ~window_s/period epochs; clipped sides need
            # fewer, but below CLIPPED_MIN_WINDOW_S the median level is not resolvable.
            min_epochs = MIN_EPOCHS_FULL
            if (t_ev - t_start) < 250.0 or (t_end - t_ev) < 250.0:
                if min(t_ev - t_start, t_end - t_ev) < CLIPPED_MIN_WINDOW_S:
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


def run(db, receivers, master_rec, master_tag, period, window_start, window_end,
        window_s=300.0, min_step=0.05, max_step=5.0, apply=True):
    """Full pre-step. apply=True writes tblClockJumpEvents, shifts tblDetectionRaw and
    invalidates the metronome tables. Returns (notes dict, jumps list)."""
    t0, t1 = naive_seconds(window_start), naive_seconds(window_end)
    con = sqlite3.connect("file:%s?mode=ro" % os.path.abspath(db).replace("\\", "/"), uri=True)
    try:
        jumps, evidence_only, unresolvable, n_events = quantify_jumps(
            con, receivers, master_rec, master_tag, period, t0, t1, window_s, min_step, max_step)
    finally:
        con.close()
    rows_shifted, affected = (0, [])
    if apply and jumps:
        rows_shifted, affected = apply_offsets(db, jumps)
    notes = {"events": n_events,
             "n_jumps": len(jumps), "evidence_only": len(evidence_only), "unresolvable": len(unresolvable),
             "rows_shifted": rows_shifted, "affected_receivers": affected,
             "window_s": window_s, "min_step_s": min_step, "max_step_s": max_step,
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
