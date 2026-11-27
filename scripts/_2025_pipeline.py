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


def clock_jumps_done(c):
    if not c._ats.get("apply_clock_jumps", False):
        return True
    return count(c.db, lp.PROGRESS_TABLE, "where Tag_ID = ?", ("__ats_clock_jumps__",)) > 0


def clock_jumps_run(c):
    """Step 1c (experimental; [ats_2025_pipeline] apply_clock_jumps = true): correct receiver clock
    jumps BEFORE clock sync (scripts/clock_jump_pre.py). Events from the per-file internal
    offset-change codes in tblDetectionRaw (OffsetChanged/CounterRestart), magnitudes from
    the master-beacon TDoA median +/- clock_jump_window_s per side, applied to
    tblDetectionRaw.seconds with the audit column jump_offset_s, recorded in
    tblClockJumpEvents. Downstream tables are dropped so the pipeline rebuilds them on the
    corrected times. Jumps without an internal code are left to the legacy auto-detect."""
    import clock_jump_pre
    ats = c._ats
    master_rec, master_tag = lp.master(c.db)
    period = float(lp.query(c.db, "select pulseRate from tblTag where Tag_ID = ?", (master_tag,)).pulseRate.iloc[0])
    notes, jumps = clock_jump_pre.run(
        str(c.db), list(dict.fromkeys(c.surface + c.deep)), master_rec, master_tag, period,
        c.window_start, c.window_end,
        window_s=float(ats.get("clock_jump_window_s", 300)),
        min_step=float(ats.get("clock_jump_min_step_s", 0.05)),
        max_step=float(ats.get("clock_jump_max_step_s", 5.0)),
        apply=True)
    if jumps:
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
    lp.clock_fix_check(c.db, c.surface,
                       segmented_clock_dbscan=bool(c._ats.get("segmented_clock_dbscan", False)))
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
    root = c._positioning.get("deep_solution", "B")
    return all(count(c.db, "tblDeepReceiver_step6_solution", "where Rec_ID = ? and solution_root = ?",
                     (receiver, root)) == 1 for receiver in c.deng)


def deep_positions_run(c):
    """Step 6: legacy Deng configured-root median for each Deng receiver beacon (deep, then static surface)."""
    rec = lp.query(c.db, "select Rec_ID, Tag_ID, X_t, Y_t, Z_t from tblReceiver").set_index("Rec_ID")
    root = c._positioning.get("deep_solution", "B")
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
    return str(c._positioning.get("adopt_deep_z", "solved")).lower()


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
    lp.clock_fix(c.db, receivers, c.work,
                 segmented_clock_dbscan=bool(c._ats.get("segmented_clock_dbscan", False)))
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


OUTSIDE_ARRAY = "outside receiver array - rejected"


def deng_settings(c):
    """[positioning] options for step 11 (all off by default = legacy behaviour):
    deng_max_receivers  – > 0: solve each transmission with only its N earliest-arriving receivers (fewer receiver
                          combinations; 20 receivers give up to 4,845 per ping, 8 give 70)
    deng_keep_deep      – with the cap, if none of the N heard first is deep, swap the Nth for the earliest deep
                          receiver that heard the ping (default true; shallow-only sets give poor Z)
    deng_hull_filter    – reject solutions more than deng_hull_margin_m (XY) outside the 3D receiver hull
                          (comment OUTSIDE_ARRAY; poor-geometry extrapolations)"""
    p = c._positioning
    return {"max_receivers": int(p.get("deng_max_receivers", 0) or 0),
            "keep_deep": bool(p.get("deng_keep_deep", True)),
            "hull_filter": bool(p.get("deng_hull_filter", False)),
            "hull_margin_m": float(p.get("deng_hull_margin_m", 5.0))}


def cap_receivers(tag_data, max_receivers, deep=()):
    """Keep each transmission's max_receivers earliest detections (by seconds_fix). With deep receivers given, a
    transmission whose kept set has none gets its latest kept detection replaced by its earliest deep detection.
    Returns (capped frame, number of transmissions given a deep receiver)."""
    df = tag_data.sort_values("seconds_fix", kind="stable")
    rank = df.groupby("transNo").cumcount()
    keep, late = df[rank < max_receivers], df[rank >= max_receivers]
    swapped = 0
    if len(deep) and len(late):
        has_deep = keep.Rec_ID.isin(deep).groupby(keep.transNo).any()
        add = late[late.transNo.isin(has_deep.index[~has_deep]) & late.Rec_ID.isin(deep)].drop_duplicates("transNo")
        if len(add):
            drop = keep.index[keep.transNo.isin(add.transNo) & (rank.loc[keep.index] == max_receivers - 1)]
            keep = pd.concat([keep.drop(drop), add]).sort_values("seconds_fix", kind="stable")
            swapped = len(add)
    return keep.reset_index(drop=True), swapped


def reject_outside_hull(out, tag, hull_eq, margin_m):
    """Mark solved rows of the tag's Deng CSVs more than margin_m (XY) outside the hull; returns rows rejected."""
    rejected = 0
    for root in ("A", "B"):
        f = Path(out) / ("%s_solution%s.csv" % (tag, root))
        if not f.exists():
            continue
        d = pd.read_csv(f, index_col=0, dtype=str, keep_default_na=False)
        found = d.comment == "solution found"
        xy = d.loc[found, ["X", "Y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        bad = ~(np.max(xy @ hull_eq[:, :2].T + hull_eq[:, 2], axis=1) <= margin_m)
        d.loc[d.index[found][bad], "comment"] = OUTSIDE_ARRAY
        d.to_csv(f)
        rejected += int(bad.sum())
    return rejected


def positions_run(c):
    """Step 11: Deng for each fish tag with the configured 3D receivers (roots A and B written per tag)."""
    out = Path(lp.fresh_folder(c.work / "positions"))
    s = deng_settings(c)
    print("DENG options: %s" % s, flush=True)
    hull_eq = None
    if s["hull_filter"]:
        from scipy.spatial import ConvexHull
        rec = lp.query(c.db, "select Rec_ID, X_t, Y_t from tblReceiver").set_index("Rec_ID")
        hull_eq = ConvexHull(rec.loc[c.receivers_3d, ["X_t", "Y_t"]].astype(float).to_numpy()).equations
    notes = {"settings": s}
    for tag in c.tags:
        pos = jsats3d.position(tag, c.receivers_3d, str(c.db), str(out), str(c.work / "figures"))
        if pos.tag_data.empty:
            raise RuntimeError("check failed: no filtered detections for %s" % tag)
        note = {}
        if s["max_receivers"] > 0:
            pos.tag_data, note["deep_swapped"] = cap_receivers(pos.tag_data, s["max_receivers"],
                                                               c.deep if s["keep_deep"] else ())
        t = time.time()
        pos.Deng()
        note["deng_s"] = round(time.time() - t, 1)
        note["A_found"] = int((pos.DengSolutionA_unfiltered.comment == "solution found").sum())
        note["B_found"] = int((pos.DengSolutionB_unfiltered.comment == "solution found").sum())
        if hull_eq is not None:
            note["outside_array"] = reject_outside_hull(out, tag, hull_eq, s["hull_margin_m"])
        print("DENG %s: %s" % (tag, note), flush=True)
        notes[tag] = note
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

    Reads [ats_2025_pipeline] bathymetry settings:
        bathy_path          – path to bathymetry XYZ/CSV/GeoTIFF
        bathy_input_crs     – CRS of the input file (default EPSG:26910)
        water_column        – if true and no bathy data, apply WSEL-only check
    """
    ats = c._ats
    bathy_path = ats.get("bathy_path")
    water_column = ats.get("water_column", False)

    if not bathy_path and not water_column:
        return {"status": "skipped", "reason": "no bathy_path, water_column = false"}

    from jsats3d import bathymetry as bth

    if bathy_path:
        bathy_path = lp.resolve_path(bathy_path)
        input_crs = ats.get("bathy_input_crs", "EPSG:26910")
        print(f"Bathymetry: loading {bathy_path} (CRS: {input_crs})")
        n = bth.load_xyz(str(c.db), bathy_path, input_crs=input_crs)
        print("Bathymetry: checking position solutions")
        bth.check_positions(str(c.db), wsel_tolerance_m=0.05)
        return {"status": "bathymetry", "points_loaded": n}
    else:
        print("Bathymetry: no bathy data; applying WSEL-only water column check")
        bth.check_positions_simple(str(c.db), water_column=True)
        return {"status": "water_column_only"}


# Step 10c (optional, [positioning] premedian_filter = true): pre-median screen of the Deng solutions of each
# transmission, so clock-jump artefacts cannot bias the per-transmission median/mean used downstream. Rejected rows
# stay in tblPositions_Deng with a 'premedian ...' comment (downstream selects comment = 'solution found').
PREMEDIAN_BOX = "premedian box - rejected"
PREMEDIAN_OUTLIER = "premedian epoch outlier - rejected"
PREMEDIAN_UNSUPPORTED = "premedian epoch unsupported - rejected"


def premedian_settings(c):
    p = c._positioning
    return {"enabled": bool(p.get("premedian_filter", False)),
            # Box gate on/off, and its Z limits on/off (off = XY-only box).
            "box": bool(p.get("premedian_box", True)),
            "box_z": bool(p.get("premedian_box_z", True)),
            # XY: within this distance outside the convex hull of the 3D receivers (PM 2026-11-16 box used 5 m).
            "box_margin_m": float(p.get("premedian_box_margin_m", 5.0)),
            # Z: receiver frame is depth below the surface (Z_t < 0); the surface is Z = 0.
            "box_z_top_m": float(p.get("premedian_box_z_top_m", 0.0)),
            "box_z_pad_m": float(p.get("premedian_box_z_pad_m", 3.0)),
            # Epoch radius: same 10 m as the project's receiver-set consensus (PM 2026-10-14).
            "radius_m": float(p.get("premedian_radius_m", 10.0)),
            "min_sets": int(p.get("premedian_min_sets", 2))}


def premedian_rejected(c):
    return count(c.db, "tblPositions_Deng", "where comment like 'premedian%'")


def premedian_done(c):
    if not premedian_settings(c)["enabled"]:
        return True
    if "tblPreMedianFilter" not in table_names(c.db) or "tblPositions_Deng" not in table_names(c.db):
        return False
    saved = lp.query(c.db, "select total_rows, rejected, settings from tblPreMedianFilter where scope = 'all'")
    return (not saved.empty and int(saved.total_rows.iloc[0]) == count(c.db, "tblPositions_Deng")
            and int(saved.rejected.iloc[0]) == premedian_rejected(c)
            and json.loads(saved.settings.iloc[0]) == premedian_settings(c))


def premedian_run(c):
    """Step 10c (optional): box gate (3D receiver hull + margin, surface to deepest receiver - pad; premedian_box /
    premedian_box_z switch it or its Z limits off; non-finite solutions are always rejected), then per
    transmission and root keep the solutions within radius_m (XY) of the densest solution; fewer than min_sets
    distinct receiver sets rejects the transmission. With the filter on, positions_dbscan uses the ping median."""
    s = premedian_settings(c)
    if not s["enabled"]:
        return {"skipped": "premedian_filter = false (positions unchanged)"}
    from scipy.spatial import ConvexHull
    con = sqlite3.connect(c.db)
    reset = con.execute("update tblPositions_Deng set comment = 'solution found' where comment like 'premedian%'").rowcount
    con.commit()
    rec = lp.query(c.db, "select Rec_ID, X_t, Y_t, Z_t from tblReceiver").set_index("Rec_ID")
    missing = [r for r in c.receivers_3d if r not in rec.index]
    if missing:
        con.close()
        raise RuntimeError("check failed: receivers_3d not in tblReceiver: %s" % missing)
    rec = rec.loc[c.receivers_3d, ["X_t", "Y_t", "Z_t"]].astype(float)
    eq = ConvexHull(rec[["X_t", "Y_t"]].to_numpy()).equations
    z_min = float(rec.Z_t.min()) - s["box_z_pad_m"]
    pos = lp.query(c.db, "select rowid as rid, Tag_ID, solution, transNo, r0, r1, r2, r3, X, Y, Z "
                         "from tblPositions_Deng where comment = 'solution found'")
    for col in ("X", "Y", "Z"):
        pos[col] = pd.to_numeric(pos[col], errors="coerce")
    xy = pos[["X", "Y"]].to_numpy(dtype=float)
    finite = np.isfinite(pos[["X", "Y", "Z"]].to_numpy(dtype=float)).all(axis=1)
    # Largest signed distance beyond a hull edge (unit normals); <= margin means inside the mitred margin band.
    beyond = np.max(np.nan_to_num(xy) @ eq[:, :2].T + eq[:, 2], axis=1)
    box_bad = ~finite
    if s["box"]:
        box_bad = box_bad | (beyond > s["box_margin_m"])
        if s["box_z"]:
            box_bad = box_bad | (pos.Z.to_numpy() > s["box_z_top_m"]) | (pos.Z.to_numpy() < z_min)
    reason = pd.Series("", index=pos.index)
    reason[box_bad] = PREMEDIAN_BOX
    keep = pos[~box_bad]
    sets = pd.Series(["+".join(r) for r in np.sort(keep[["r0", "r1", "r2", "r3"]].astype(str).to_numpy(), axis=1)],
                     index=keep.index)
    r2 = s["radius_m"] ** 2
    for _, g in keep.groupby(["Tag_ID", "solution", "transNo"], sort=False):
        p = g[["X", "Y"]].to_numpy(dtype=float)
        d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=-1)
        near = d2 <= r2
        n = near.sum(axis=1)
        ties = np.flatnonzero(n == n.max())
        best = ties[np.argmin([d2[i][near[i]].sum() for i in ties])]
        inside = near[best]
        if sets.loc[g.index[inside]].nunique() < s["min_sets"]:
            reason[g.index] = PREMEDIAN_UNSUPPORTED
        else:
            reason[g.index[~inside]] = PREMEDIAN_OUTLIER
    flagged = reason[reason != ""]
    con.executemany("update tblPositions_Deng set comment = ? where rowid = ?",
                    list(zip(flagged.to_numpy(), pos.loc[flagged.index, "rid"].astype(int).tolist())))
    # Audit: per tag/root and per receiver (share of the receiver's solutions that were rejected).
    pos["reason"] = reason
    rows = []
    for (tag, root), g in pos.groupby(["Tag_ID", "solution"]):
        rows.append({"scope": "tag_root", "Tag_ID": tag, "solution": root, "Rec_ID": None, "found_before": len(g),
                     "box": int((g.reason == PREMEDIAN_BOX).sum()), "outlier": int((g.reason == PREMEDIAN_OUTLIER).sum()),
                     "unsupported": int((g.reason == PREMEDIAN_UNSUPPORTED).sum()),
                     "kept_transmissions": int(g[g.reason == ""].transNo.nunique()),
                     "transmissions_before": int(g.transNo.nunique())})
    members = pos[["r0", "r1", "r2", "r3"]].astype(str)
    for r in sorted(set(members.to_numpy().ravel()) - {"nan", "None"}):
        g = pos[members.eq(r).any(axis=1)]
        rows.append({"scope": "receiver", "Tag_ID": None, "solution": None, "Rec_ID": r, "found_before": len(g),
                     "box": int((g.reason == PREMEDIAN_BOX).sum()), "outlier": int((g.reason == PREMEDIAN_OUTLIER).sum()),
                     "unsupported": int((g.reason == PREMEDIAN_UNSUPPORTED).sum()),
                     "kept_transmissions": None, "transmissions_before": None})
    summary = pd.DataFrame(rows)
    summary["rejected_share"] = (summary.box + summary.outlier + summary.unsupported) / summary.found_before.clip(lower=1)
    total = int(con.execute("select count(*) from tblPositions_Deng").fetchone()[0])
    summary = pd.concat([summary, pd.DataFrame([{"scope": "all", "found_before": len(pos), "box": int(box_bad.sum()),
                                                  "outlier": int((reason == PREMEDIAN_OUTLIER).sum()),
                                                  "unsupported": int((reason == PREMEDIAN_UNSUPPORTED).sum())}])],
                        ignore_index=True)
    summary["total_rows"] = total
    summary["rejected"] = len(flagged)
    summary["settings"] = json.dumps(s)
    summary.to_sql("tblPreMedianFilter", con, if_exists="replace", index=False)
    # Downstream products were built from the unfiltered solutions.
    con.execute("drop table if exists tblPositions_DengDBSCAN")
    con.commit()
    con.close()
    (plot_dbscan_folder(c) / "item12_pipeline_summary.csv").unlink(missing_ok=True)
    print("PRE-MEDIAN FILTER: %d of %d found solutions rejected (box %d, epoch outlier %d, unsupported %d); "
          "%d earlier premedian flags reset; settings %s" % (len(flagged), len(pos), int(box_bad.sum()),
          int((reason == PREMEDIAN_OUTLIER).sum()), int((reason == PREMEDIAN_UNSUPPORTED).sum()), reset, s), flush=True)
    worst = summary[summary.scope == "receiver"].sort_values("rejected_share", ascending=False).head(5)
    print("PRE-MEDIAN FILTER: highest receiver rejection shares: %s" % ", ".join(
        "%s %.0f%%" % (r, 100 * v) for r, v in zip(worst.Rec_ID, worst.rejected_share)), flush=True)
    return {"rejected": len(flagged), "found_before": len(pos), "box": int(box_bad.sum()),
            "outlier": int((reason == PREMEDIAN_OUTLIER).sum()), "unsupported": int((reason == PREMEDIAN_UNSUPPORTED).sum()),
            "by_tag_root": summary[summary.scope == "tag_root"][["Tag_ID", "solution", "found_before", "rejected_share",
                                                                  "transmissions_before", "kept_transmissions"]].to_dict("records"),
            "receiver_rejected_share": {r: round(float(v), 3) for r, v in zip(worst.Rec_ID, worst.rejected_share)},
            "settings": s}






def positions_2d_done(c):
    legacy_pl = c.run.get("legacy_pipeline", {})
    if not legacy_pl.get("known_depth_diagnostics", False):
        return True
    known_depth_tags = [tag for tag in c.tags if tag in c.fixed_z_2d]
    return not known_depth_tags or "tblPositions_Deng2D" in table_names(c.db)


def positions_2d_run(c):
    """Legacy Deng2D (every receiver set of three, roots A and B) with fish Z fixed per tag, over receivers_2d."""
    legacy_pl = c.run.get("legacy_pipeline", {})
    if not legacy_pl.get("known_depth_diagnostics", False):
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
    legacy_pl = c.run.get("legacy_pipeline", {})
    return not legacy_pl.get("known_depth_diagnostics", False) or (plot_drag_2d_folder(c) / "item12_pipeline_summary.csv").exists()


def plot_drag_2d_run(c):
    """Optional: scores and plots Deng2D (fish Z fixed) root-A centroids against the drag GPS."""
    legacy_pl = c.run.get("legacy_pipeline", {})
    if not legacy_pl.get("known_depth_diagnostics", False):
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
    return bool(c._positioning.get("positions_dbscan", False))


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
    [positioning] dbscan_mode 'distance' (default): DBSCAN on per-transmission XY.
    'acceleration': DBSCAN on XY, depth * dbscan_depth_weight and horizontal acceleration * dbscan_accel_scale (m per m/s^2)."""
    if not positions_dbscan_enabled(c):
        return {"skipped": "positions_dbscan = false (legacy steps unchanged)"}
    from sklearn.cluster import DBSCAN
    pos_cfg = c._positioning
    mode = str(pos_cfg.get("dbscan_mode", "distance")).lower()
    if mode not in ("distance", "acceleration"):
        raise ValueError("dbscan_mode must be 'distance' or 'acceleration', got %r" % mode)
    eps = float(pos_cfg.get("dbscan_eps_m", 8.0 if mode == "acceleration" else 5.0))
    min_samples = int(pos_cfg.get("dbscan_min_samples", 5))
    accel_scale = float(pos_cfg.get("dbscan_accel_scale", 5.0))
    depth_weight = float(pos_cfg.get("dbscan_depth_weight", 1.0))
    gap_s = float(pos_cfg.get("dbscan_accel_gap_s", 12.0))
    # Only the columns the screen needs (the full table can hold tens of millions of rows - 100-fish run MemoryError).
    root = lp.query(c.db, "select Tag_ID, transNo, X, Y, Z, ToA, T01 from tblPositions_Deng "
                          "where comment = 'solution found' and solution = 'B'")
    if mode == "acceleration":
        if "T01" not in root:
            raise ValueError("acceleration mode needs T01 in tblPositions_Deng")
        root = root.assign(**{col: pd.to_numeric(root[col], errors="coerce") for col in ("X", "Y", "Z", "ToA", "T01")})
        root = root.assign(t0=root.ToA - root.T01)
        root = root[np.isfinite(root[["X", "Y", "Z", "t0"]]).all(axis=1) & root[["X", "Y", "Z"]].abs().lt(1000).all(axis=1)]
    kept, notes = [], {}
    # With the pre-median filter on, the ping position is the median of the surviving solutions (robust); otherwise
    # the original mean (unchanged behaviour).
    agg = "median" if premedian_settings(c)["enabled"] else "mean"
    for tag, rows in root.groupby("Tag_ID"):
        if mode == "distance":
            ping = rows.groupby("transNo")[["X", "Y"]].agg(agg)
            features = ping.to_numpy(dtype=float)
        else:
            ping = rows.groupby("transNo").agg(X=("X", agg), Y=("Y", agg), Z=("Z", agg), t0=("t0", "median")).sort_values("t0")
            accel = ping_acceleration(ping.t0.to_numpy(), ping[["X", "Y"]].to_numpy(), gap_s)
            accel = np.where(np.isnan(accel), np.nanmedian(accel) if np.isfinite(accel).any() else 0.0, accel)
            features = np.column_stack([ping.X, ping.Y, depth_weight * ping.Z, accel_scale * accel])
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)
        kept_trans = set(ping.index[labels != -1])
        notes[tag] = {"mode": mode, "transmissions": int(len(ping)), "kept": int(len(kept_trans)), "eps_m": eps, "min_samples": min_samples}
        kept.extend((tag, float(t)) for t in kept_trans)
    del root
    # Copy every 'solution found' row (both roots, all columns) of the kept transmissions inside SQLite.
    con = sqlite3.connect(c.db)
    con.execute("drop table if exists temp.dbscan_kept")
    con.execute("create temp table dbscan_kept (Tag_ID TEXT, transNo REAL, primary key (Tag_ID, transNo))")
    con.executemany("insert into temp.dbscan_kept values (?, ?)", kept)
    con.execute("drop table if exists tblPositions_DengDBSCAN")
    con.execute("create table tblPositions_DengDBSCAN as select p.* from tblPositions_Deng p "
                "join temp.dbscan_kept k on p.Tag_ID = k.Tag_ID and cast(p.transNo as real) = k.transNo "
                "where p.comment = 'solution found'")
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


STEPS = [("prepare", "1b", prepare_done, prepare_run), ("clock_jumps", "1c", clock_jumps_done, clock_jumps_run),
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
         ("premedian", "10c", premedian_done, premedian_run),
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
    # Support both new config sections ([positioning], [ats_2025_pipeline]) and legacy ([legacy]).
    positioning = run.get("positioning", {})
    ats_settings = run.get("ats_2025_pipeline", {})
    legacy = run.get("legacy", {})
    # Merge: new sections take priority, fall back to legacy for backward compat.
    merged_pos = {}
    merged_pos.update(legacy)
    merged_pos.update(positioning)
    merged_ats = {}
    merged_ats.update(ats_settings)
    study, selection = run["study"], run["selection"]
    surface, deep = list(merged_pos.get("surface_receivers") or []), list(merged_pos.get("deep_receivers") or [])
    receivers_3d = list(merged_pos.get("receivers_3d") or [])
    master_rec, _ = lp.master(db)
    if master_rec not in surface:
        raise SystemExit("master receiver %s must be in surface_receivers" % master_rec)
    study_tags = lp.query(db, "select Tag_ID from tblTag where TagType = 'study' and pulseRate is not null").Tag_ID.tolist()
    tags = [t for t in (selection.get("tags") or study_tags) if t in study_tags]
    no_rate = [t for t in (selection.get("tags") or []) if t not in study_tags]
    if no_rate:
        print("WARNING: %d selected tag(s) skipped - no pulse rate (run file, workbook or calculated from the data): %s"
              % (len(no_rate), no_rate), flush=True)
    if not tags:
        raise SystemExit("no study tags with a pulse rate in %s" % db)
    work = lp.resolve_path(db.parent / ("%s_legacy" % db.stem))
    work.mkdir(parents=True, exist_ok=True)
    deng = deep + ([r for r in receivers_3d if r in surface and r != master_rec]
                   if merged_pos.get("deng_surface", False) else [])  # static only; CFD floats never targets
    deng_tags = {r: lp.query(db, "select Tag_ID from tblReceiver where Rec_ID = ?", (r,)).Tag_ID.iloc[0] for r in deng}
    drag_plots = not getattr(args, "no_drag_plots", False)
    return SimpleNamespace(run=run, paths=paths, db=db, work=work, surface=surface, deep=deep, receivers_3d=receivers_3d,
                           receivers_2d=list(merged_pos.get("receivers_2d") or []),
                           fixed_z_2d=dict(merged_pos.get("fixed_z_2d_by_tag") or {}),
                           review=lp.REPO / "output" / "2025_review" / db.stem,
                           reference=[r for r in receivers_3d if r not in deep],
                           reference_2d=[r for r in (merged_pos.get("receivers_2d") or receivers_3d) if r not in deep],
                           tags=tags, deng=deng, deng_tags=deng_tags,
                           window_start=str(study["synch_time_start"]), window_end=str(study["synch_time_end"]),
                           drag_plots=drag_plots,
                           _positioning=merged_pos, _ats=merged_ats)


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
    args = parser.parse_args(argv)
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
