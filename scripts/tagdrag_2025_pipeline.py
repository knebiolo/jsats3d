"""2025 tag drag workflow after the build: paper steps 2-11 on a legacy-format database, one command, resumable.

    python scripts/run_data.py config/run_data_tagdrag_ent.toml                   step 1 (parse raw files, study parameters)
    python scripts/tagdrag_2025_pipeline.py config/run_data_tagdrag_ent.toml      everything after step 1
    options: --db PATH (other database, own work folder)  --only NAME ...  --from NAME  --to NAME  --list
             --init-from DB [--swap REC_A REC_B]  what-if copy of a finished database, optionally with two hydrophone positions exchanged

Run order and names (paper step in brackets):
    prepare (1b)  metronome (2-3)  clock_surface (4)  deep_beacons (5)  deep_positions (6)  adopt_deep (6b)  deep_clocks (7)
    fish_multipath (8)  speed_of_sound (9)  positions (11)  export (10)  positions_2d (optional)  plot_drag (12)
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

warnings.filterwarnings("ignore")
jsats3d = lp.jsats3d

# Forebay minus collector-entrance mean(S, N): median over the overlap 2025-06-17 00:05-11:10 (journal 2026-10-02).
WSE_OFFSET_FT = 0.8905
# Step 8 screen: a first arrival later than this against the rolling median of its own (tag, receiver) series is a
# suspected reflection (window in transmissions). The boat moves range about 1.2 ms per ping; reflections sit several ms late.
SUSPECT_WINDOW = 9
SUSPECT_LATE_S = 0.003


# Drag truth is used only after Deng has solved XYZ.
DRAG_LINES = ("ENT-01", "ENT-05")


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
            raise RuntimeError("no legacy root-%s solutions for deep receiver %s" % (root, r))
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


def adopt_deep_done(c):
    if "tblDeepReceiver_step6_solution" not in table_names(c.db):
        return False
    sol = lp.query(c.db, "select Rec_ID, X, Y, Z from tblDeepReceiver_step6_solution").set_index("Rec_ID")
    rec = lp.query(c.db, "select Rec_ID, X_t, Y_t, Z_t from tblReceiver").set_index("Rec_ID")
    solved = {r: [sol.at[r, "X"], sol.at[r, "Y"], sol.at[r, "Z"] if r in c.deep else rec.at[r, "Z_t"]] for r in sol.index}
    return all(r in solved and np.allclose(rec.loc[r, ["X_t", "Y_t", "Z_t"]].astype(float).values, solved[r], atol=1e-6)
                                              for r in c.deng)


def adopt_deep_run(c):
    """Step 6b (legacy phase 2): write the solved deep positions into tblReceiver, then invalidate every output that
    was computed with the config positions (deep clocks, fish multipath, positions, export, plots)."""
    sol = lp.query(c.db, "select Rec_ID, X, Y, Z from tblDeepReceiver_step6_solution").set_index("Rec_ID")
    cfg = lp.query(c.db, "select Rec_ID, X_t, Y_t, Z_t from tblReceiver").set_index("Rec_ID")
    notes = {}
    for r in c.deng:
        x, y = (float(sol.at[r, k]) for k in ("X", "Y"))
        # surface Z stays configured; Deng XY only
        z = float(sol.at[r, "Z"]) if r in c.deep else float(cfg.at[r, "Z_t"])
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
    for t in ("tblPositions_Deng", "tblPositions_Deng2D", "tblPositions_Deng3DConsensus",
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
                (c.review / ("item12_pipeline3d_%s_%s%s.png" % (line, tag, suffix))).unlink(missing_ok=True)
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
               "drop table if exists tblPositions_Deng", "drop table if exists tblPositions_Deng2D")
    for folder in ("positions", "positions_2d"):
        shutil.rmtree(c.work / folder, ignore_errors=True)
    for line in DRAG_LINES:
        (c.review / ("%s_positions.png" % line.replace("-", ""))).unlink(missing_ok=True)
        for tag in ("FC36", "FFD3"):
            for suffix in ("", "_full_range"):
                (c.review / ("item12_pipeline3d_%s_%s%s.png" % (line, tag, suffix))).unlink(missing_ok=True)
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
        con = sqlite3.connect(c.db)
        try:
            for table in ("tblDetectionFilterPrimary", "tblDetectionFilterSecondary"):
                if table in table_names(c.db):
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
    return False


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
    return "tblPositions_Deng" in table_names(c.db) or any((c.work / "positions").glob("*.csv"))


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
    return all((c.review / ("item12_pipeline3d_%s_%s.png" % (line, tag))).exists()
               for line in DRAG_LINES for tag in ("FC36", "FFD3"))


def plot_drag_run(c):
    """Step 12: four black-GPS/blue-Deng XYZ plots; truth is used only for scoring."""
    if not {"FC36", "FFD3"} <= set(c.tags):
        return {"skipped": "ENT truth is available only for FC36 and FFD3"}
    gps_file = Path(c.paths["config_xlsx"]).parents[1] / "5_array_testing" / "array_testing_drag_GPS.csv"
    cmd = [sys.executable, str(HERE / "ent_analysis_2025.py"), "--db", str(c.db),
           "--out", c.db.stem, "--only", "12", "--gps", str(gps_file)]
    result = subprocess.run(cmd, cwd=lp.REPO)
    if result.returncode != 0 or not plot_drag_done(c):
        raise RuntimeError("step 12 failed to generate the four legacy Deng comparison plots")
    return {"folder": str(c.review), "blue_source": "tblPositions_Deng root B mean XYZ",
            "gps_and_tag_depth_used_for_solving": False}


def positions_dbscan_enabled(c):
    return bool(c.run["legacy"].get("positions_dbscan", False))


def positions_dbscan_done(c):
    return not positions_dbscan_enabled(c) or "tblPositions_DengDBSCAN" in table_names(c.db)


def positions_dbscan_run(c):
    """Optional screen: DBSCAN on each tag's root-B XY per transmission; keeps every root row of transmissions in clusters."""
    if not positions_dbscan_enabled(c):
        return {"skipped": "positions_dbscan = false (legacy steps unchanged)"}
    from sklearn.cluster import DBSCAN
    eps = float(c.run["legacy"].get("dbscan_eps_m", 5.0))
    min_samples = int(c.run["legacy"].get("dbscan_min_samples", 5))
    pos = lp.query(c.db, "select * from tblPositions_Deng where comment = 'solution found'")
    root = pos[pos.solution == "B"]
    kept, notes = [], {}
    for tag, rows in root.groupby("Tag_ID"):
        xy = rows.groupby("transNo")[["X", "Y"]].mean()
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy.to_numpy(dtype=float))
        kept_trans = set(xy.index[labels != -1])
        notes[tag] = {"transmissions": int(len(xy)), "kept": int(len(kept_trans)), "eps_m": eps, "min_samples": min_samples}
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
           "--only", "12", "--gps", str(gps_file), "--table", "tblPositions_DengDBSCAN", "--plots", "3D_legacy_B"]
    result = subprocess.run(cmd, cwd=lp.REPO)
    if result.returncode != 0 or not plot_drag_dbscan_done(c):
        raise RuntimeError("DBSCAN plot step did not write its summary")
    return {"folder": str(plot_dbscan_folder(c)), "pngs": sorted(p.name for p in plot_dbscan_folder(c).glob("item12_pipeline3d_*.png"))}


STEPS = [("prepare", "1b", prepare_done, prepare_run), ("metronome", "2-3", metronome_done, metronome_run),
         ("clock_surface", "4", clock_surface_done, clock_surface_run),
         ("deep_beacons", "5", deep_beacons_done, deep_beacons_run),
         ("deep_positions", "6", deep_positions_done, deep_positions_run),
         ("adopt_deep", "6b", adopt_deep_done, adopt_deep_run),
         ("deep_clocks", "7", deep_clocks_done, deep_clocks_run),
         ("fish_multipath", "8", fish_multipath_done, fish_multipath_run),
         ("speed_of_sound", "9", speed_of_sound_done, speed_of_sound_run),
         ("positions", "11", positions_done, positions_run), ("export", "10", export_done, export_run),
         ("positions_2d", "11b", positions_2d_done, positions_2d_run),
         ("plot_drag", "12", plot_drag_done, plot_drag_run),
         ("positions_dbscan", "11c", positions_dbscan_done, positions_dbscan_run),
         ("plot_drag_dbscan", "12b", plot_drag_dbscan_done, plot_drag_dbscan_run)]


# ----------------------------------------------------------------------------------------------- driver

def build_context(args):
    run, paths = lp.read_run_file(args.run_file)
    db = lp.resolve_path(args.db) if args.db else paths["output_db"]
    if not db.exists():
        raise SystemExit("database %s not found; run scripts/run_data.py with this run file first" % db)
    legacy, study, selection = run["legacy"], run["study"], run["selection"]
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
    return SimpleNamespace(run=run, paths=paths, db=db, work=work, surface=surface, deep=deep, receivers_3d=receivers_3d,
                           receivers_2d=list(legacy.get("receivers_2d") or []), fixed_z_2d=dict(legacy.get("fixed_z_2d_by_tag") or {}),
                           review=lp.REPO / "output" / "2025_review" / db.stem,
                           reference=[r for r in receivers_3d if r not in deep], reference_2d=[r for r in (legacy.get("receivers_2d") or receivers_3d) if r not in deep], tags=tags, deng=deng, deng_tags=deng_tags,
                           window_start=str(study["synch_time_start"]), window_end=str(study["synch_time_end"]))


def init_whatif(source, target, swaps):
    """New database from a finished one: derived tables dropped, tblReceiver back at config, then the listed hydrophone positions swapped.

    Only coordinates move (X, Y, Z, X_t, Y_t, Z_t, easting, northing); clock, tag and mount stay with the receiver id.
    tblReceiver_initial keeps the config rows and tblReceiverSwap records what was done."""
    if target.exists():
        raise SystemExit("%s exists; --init-from only creates a new database" % target)
    shutil.copy2(source, target)
    con = sqlite3.connect(target)
    for table in lp.DERIVED_TABLES + ["tblPositions_Deng", "tblPositions_Deng2D", "tblPositions_Deng3DConsensus",
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
    print("Database %s\nTags %s | master/anchor from run file | window %s to %s" % (c.db, c.tags, c.window_start, c.window_end))
    failed = None
    for name, label, done, run in STEPS:
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
            print("FAILED: %s: %s\n%s" % (type(error).__name__, error, "\n".join(trace.splitlines()[-8:])), flush=True)
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
