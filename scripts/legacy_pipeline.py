"""Run the preserved legacy jsats3d workflow on a legacy-format project database.

The workflow and scientific method remain legacy-compatible, but execute in the
active shared environment. Compatibility fixes live in jsats3d.py so 2019 and
2025 use one runtime and one set of core functions.
Called by scripts/run_data.py; can also be run directly:

    python scripts/legacy_pipeline.py import-2019 config/run_data_2019.toml
    python scripts/legacy_pipeline.py process     config/run_data.toml

import-2019  The preserved Teknologic importer for 2019 data.
process      Nebiolo & Meyer (2021) workflow, calling jsats3d functions in Kevin's order:
             1 metronome (beacon_epoch, multipath_2, multipath_classifier) on the master beacon
             2 clock fix of the surface receivers (clock_fix, epoch_fix_data_management)
             3 deep receivers: beacon multipath, position.Deng, median X/Y/Z -> tblReceiver X_t/Y_t/Z_t
             4 metronome + clock fix again with all receivers
             5 study tags: multipath, position.Deng, positions_data_management -> tblPositions_Deng
The driver only orders calls, manages scratch folders and resets derived tables between phases.
Kevin's own driver scripts are not used: they hold his paths and pass arguments jsats3d does not accept.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # legacy code calls plt.show() for every receiver

import argparse
import shutil
import sqlite3
import sys
import tempfile
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 legacy env
    import tomli as tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import jsats3d  # noqa: E402  (legacy core, unchanged from Kevin's 7e3b90e)

warnings.filterwarnings("ignore")

DERIVED_TABLES = [
    "tblMetronomeUnfiltered", "tblMetronomeFiltered", "tblMetronomeSecondFiltered",
    "tblDetectionClockFixed", "tblDetectionFilterPrimary", "tblDetectionFilterSecondary",
]
RECEIVER_DTYPES = {"Rec_ID": str, "Type": str, "Tag_ID": str, "Ref_Elev": str, "X": np.float64, "Y": np.float64,
                   "Z": np.float64, "X_t": np.float64, "Y_t": np.float64, "Z_t": np.float64}


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def read_run_file(path):
    with open(path, "rb") as stream:
        run = tomllib.load(stream)
    paths = {name: resolve_path(value) for name, value in run.get("paths", {}).items() if str(value).strip()}
    return run, paths


def blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def fresh_folder(path):
    """Legacy *_data_management functions load and delete every file in a folder: give each step its own."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return str(path)


def execute(db, *statements):
    connection = sqlite3.connect(db)
    try:
        for statement in statements:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


def index_table(db, table, name, columns):
    """Add a query index only when a derived table exists."""
    connection = sqlite3.connect(db)
    try:
        exists = connection.execute(
            "select 1 from sqlite_master where type = 'table' and name = ?", (table,)
        ).fetchone()
        if exists:
            connection.execute("create index if not exists %s on %s (%s)" %
                               (name, table, ", ".join(columns)))
            connection.commit()
    finally:
        connection.close()


def query(db, sql, params=()):
    connection = sqlite3.connect(db)
    try:
        return pd.read_sql(sql, connection, params=params)
    finally:
        connection.close()


def widen_table(db, table, folder, extra=()):
    """Give `table` every column found in the folder's CSVs before a legacy *_data_management load.

    Legacy loaders append CSVs in os.listdir order, and the first file fixes the table columns. Receivers
    that take different multipath_classifier branches (supervised, unsupervised, the excluded host) write
    different columns, so a load fails whenever a narrower file happens to come first. Rows are unchanged.
    """
    columns = []
    for path in sorted(Path(folder).glob("*.csv")):
        for column in pd.read_csv(path, nrows=0).columns:
            if column not in columns:
                columns.append(column)
    if not columns:
        return
    columns += [c for c in extra if c not in columns]
    connection = sqlite3.connect(db)
    try:
        existing = [row[1] for row in connection.execute("PRAGMA table_info(%s)" % table)]
        if not existing:
            connection.execute("CREATE TABLE %s (%s)" % (table, ", ".join('"%s"' % c for c in columns)))
        for column in columns:
            if existing and column not in existing:
                connection.execute('ALTER TABLE %s ADD COLUMN "%s"' % (table, column))
        connection.commit()
    finally:
        connection.close()


# ---------------------------------------------------------------- 2019 Teknologic import

def interpolated_temperature_2019(temp):
    """Build the 2019 mean temperature series from each location-depth profile."""
    temp = temp.copy()
    temp["time_stamp"] = pd.to_datetime(temp.meas_dt)
    temp["loc_dep_id"] = temp.location + "-" + temp.depth_ft.astype(str)
    if "temp_celcius" in temp.columns:
        temp["temperature_c"] = pd.to_numeric(temp.temp_celcius, errors="raise")
    elif "temp_f" in temp.columns:
        temp["temperature_c"] = (pd.to_numeric(temp.temp_f, errors="raise") - 32) * 5. / 9.
    else:
        raise ValueError("2019 temperature input needs temp_celcius or temp_f")
    interpolators, lows, highs = {}, [], []
    for loc in temp.loc_dep_id.unique():
        loc_dat = temp[temp.loc_dep_id == loc].sort_values("time_stamp")
        loc_dat["seconds"] = pd.DatetimeIndex(loc_dat.time_stamp).as_unit('ns').astype(np.int64) / 1.0e9
        loc_dat = loc_dat.drop_duplicates("seconds", keep="first").set_index("seconds", drop=False).dropna()
        interpolators[loc] = interp1d(loc_dat.seconds.values, loc_dat.temperature_c.values, kind="linear",
                                      bounds_error=False, fill_value=np.nan)
        lows.append(loc_dat.seconds.min())
        highs.append(loc_dat.seconds.max())
    epoch_range = np.linspace(min(lows) - 0.1, max(highs) + 0.1, 10000)
    means = []
    for timestamp in epoch_range:
        values = np.asarray([f(timestamp) for f in interpolators.values()], dtype=float)
        means.append(float(np.nanmean(values)) if np.isfinite(values).any() else np.nan)
    return pd.DataFrame({"timeStamp": pd.to_datetime(epoch_range, unit="s"), "C": means})


def import_2019(run, paths):
    study = run.get("study", {})
    db = paths["output_db"]
    db.parent.mkdir(parents=True, exist_ok=True)
    jsats3d.create_project_db(str(db.parent), db.name)
    jsats3d.set_study_parameters(study.get("utc_conv"), study.get("bm_elev"), study.get("bm_elev_units"),
                                 study.get("output_units"), study.get("master_receiver"),
                                 study.get("synch_time_start"), study.get("synch_time_end"), str(db))
    receivers = pd.read_csv(paths["receiver_csv"], dtype=RECEIVER_DTYPES)
    jsats3d.study_data_import(pd.read_csv(paths["tag_csv"]), str(db), "tblTag")
    jsats3d.study_data_import(receivers, str(db), "tblReceiver")
    jsats3d.study_data_import(pd.read_csv(paths["wsel_csv"]), str(db), "tblWSEL")
    temp = pd.read_csv(paths["temp_csv"])
    jsats3d.study_data_import(temp, str(db), "tblTemp")
    connection = sqlite3.connect(db)
    interpolated_temperature_2019(temp).to_sql("tblInterpolatedTemp", con=connection, if_exists="replace")
    connection.close()
    print("Project database set up: %s" % db)
    for rec in receivers.Rec_ID:
        folder = paths["raw_root"] / rec
        if not folder.is_dir():
            print("WARNING: no raw folder for receiver %s (%s)" % (rec, folder))
            continue
        jsats3d.acoustic_data_import(rec, "Teknologic", str(folder), str(db))
        print("Imported receiver %s" % rec)


# ---------------------------------------------------------------- processing

def master(db):
    params = query(db, "select masterReceiver from tblStudyParameters")
    if params.empty or blank(params.masterReceiver.iloc[0]):
        raise ValueError("tblStudyParameters.masterReceiver is empty; set [study] master_receiver")
    rec = params.masterReceiver.iloc[0]
    tag = query(db, "select Tag_ID from tblReceiver where Rec_ID = ?", (rec,))
    if tag.empty or blank(tag.Tag_ID.iloc[0]):
        raise ValueError("Master receiver %s has no beacon Tag_ID in tblReceiver" % rec)
    return rec, tag.Tag_ID.iloc[0]


def metronome(db, tag, work, method):
    """metronome.py: enumerate the master beacon, rank multipath, then the secondary classifier."""
    print("Metronome: beacon %s" % tag)
    epoch = jsats3d.beacon_epoch(tag, str(db), str(work))
    epoch.host_receiver_enumeration()
    epoch.adjacent_receiver_enumeration()
    jsats3d.multipath_2(jsats3d.multipath_data_object(tag, str(db), str(work), metronome=True))
    classified = fresh_folder(work / "metronome_classifier")
    jsats3d.multipath_classifier(tag, str(db), classified, metronome=True, method=method)
    widen_table(db, "tblMetronomeSecondFiltered", classified)
    jsats3d.multipath_data_management(classified, str(db), metronome=True)
    index_table(db, "tblMetronomeUnfiltered", "idx_metronome_rec_tag_seconds", ("Rec_ID", "Tag_ID", "seconds"))
    index_table(db, "tblMetronomeFiltered", "idx_metronome_filtered_rec_tag_seconds", ("Rec_ID", "Tag_ID", "seconds"))
    index_table(db, "tblMetronomeSecondFiltered", "idx_metronome_second_rec_tag_seconds", ("Rec_ID", "Tag_ID", "seconds"))


def clock_fix(db, receivers, work):
    """clock_fix_serial.py for every receiver in the list, then load tblDetectionClockFixed."""
    scratch = fresh_folder(work / "clock_fix_scratch")
    figures = work / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for rec in receivers:
        print("Clock fix: receiver %s" % rec)
        jsats3d.clock_fix(jsats3d.clock_fix_object(rec, receivers, str(db), scratch, str(figures)))
    widen_table(db, "tblDetectionClockFixed", scratch)
    jsats3d.epoch_fix_data_management(scratch, str(db))


def clock_fix_check(db, receivers):
    """Run clock fix only when it cannot append over existing clock-fixed results."""
    tables = query(db, "select name from sqlite_master where type = 'table'").name.tolist()
    if "tblDetectionClockFixed" in tables:
        raise ValueError("tblDetectionClockFixed already exists; refusing to append duplicate clock-fix results")
    master_rec, master_tag = master(db)
    selected = list(dict.fromkeys([master_rec] + receivers))
    print("Clock-fix check: master %s (beacon %s); receivers %s" % (master_rec, master_tag, selected))
    with tempfile.TemporaryDirectory(prefix="jsats3d_clock_fix_") as temporary:
        clock_fix(db, selected, Path(temporary))
    if "tblDetectionClockFixed" not in query(db, "select name from sqlite_master where type = 'table'").name.tolist():
        print("tblDetectionClockFixed was not created")
        return
    stats = query(db, "select Rec_ID, count(*) as rows_n, min(seconds_residual) as res_min, "
                      "max(seconds_residual) as res_max, avg(abs(seconds_residual)) as res_abs_mean "
                      "from tblDetectionClockFixed group by Rec_ID")
    print("\ntblDetectionClockFixed (seconds_residual, seconds):")
    print(stats.to_string(index=False))
    for rec in selected:
        values = query(db, "select seconds_residual from tblDetectionClockFixed where Rec_ID = ?", (rec,))
        if values.empty:
            print("WARNING: no clock-fixed rows for %s" % rec)
            continue
        residual_ms = values.seconds_residual.abs() * 1000.0
        print("%-6s |residual| ms: median %.4f  p95 %.4f  over 0.5 ms: %s of %s"
              % (rec, residual_ms.median(), residual_ms.quantile(0.95),
                 int((residual_ms > 0.5).sum()), len(residual_ms)))


def tag_multipath(db, tag, work, method):
    """mulitpath.py: primary ranking then secondary classifier. False when the tag has no clock-fixed data."""
    primary = fresh_folder(work / "multipath_primary")
    data = jsats3d.multipath_data_object(tag, str(db), primary)
    if data.empty:
        print("WARNING: tag %s has no clock-fixed detections; skipped" % tag)
        return False
    jsats3d.multipath_2(data)
    widen_table(db, "tblDetectionFilterPrimary", primary)
    jsats3d.multipath_data_management(primary, str(db), primary=True)
    secondary = fresh_folder(work / "multipath_secondary")
    jsats3d.multipath_classifier(tag, str(db), secondary, method=method)
    widen_table(db, "tblDetectionFilterSecondary", secondary)
    jsats3d.multipath_data_management(secondary, str(db), primary=False)
    # Detections outside every host ping window keep transNo NULL; position.Deng then writes a 'nan'
    # transmission row that pd.to_numeric cannot parse and the run stops. They cannot be positioned anyway.
    connection = sqlite3.connect(db)
    dropped = connection.execute("DELETE FROM tblDetectionFilterSecondary WHERE Tag_ID = ? AND transNo IS NULL",
                                 (tag,)).rowcount
    connection.commit()
    connection.close()
    if dropped:
        print("Tag %s: %s detections outside every ping window (transNo empty) removed before Deng" % (tag, dropped))
    return True


def deng(db, tag, receivers, out, figures):
    pos = jsats3d.position(tag, receivers, str(db), str(out), str(figures))
    if pos.tag_data.empty:
        print("WARNING: tag %s has no filtered detections at these receivers; no positions" % tag)
        return None
    pos.Deng()
    return pos


def position_deep_receivers(db, surface, deep, work, method, solution):
    """coordinate_with_Deng.py: position each deep receiver's own beacon, write the median to X_t/Y_t/Z_t."""
    out = Path(fresh_folder(work / "deep_receivers"))
    for rec in deep:
        tag = query(db, "select Tag_ID from tblReceiver where Rec_ID = ?", (rec,)).Tag_ID.iloc[0]
        print("Deep receiver %s: beacon %s" % (rec, tag))
        if not tag_multipath(db, tag, work, method):
            continue
        pos = deng(db, tag, surface, out, work / "figures")
        if pos is None:
            continue
        sols = pos.DengSolutionA_unfiltered if solution == "A" else pos.DengSolutionB_unfiltered
        sols = sols[sols.comment == "solution found"]
        if sols.empty:
            print("WARNING: no Deng solutions for %s; X_t/Y_t/Z_t unchanged" % rec)
            continue
        x, y, z = (float(sols[c].median()) for c in ("X", "Y", "Z"))
        execute(db, "update tblReceiver set X_t = %r, Y_t = %r, Z_t = %r where Rec_ID = '%s'" % (x, y, z, rec))
        print("Deep receiver %s: median of %s solution-%s positions -> X_t %.3f, Y_t %.3f, Z_t %.3f"
              % (rec, len(sols), solution, x, y, z))


def process(run, paths):
    legacy = run.get("legacy", {})
    db = paths["output_db"]
    method = legacy.get("method", "KNN")
    surface = list(legacy.get("surface_receivers") or [])
    deep = list(legacy.get("deep_receivers") or [])
    solution = legacy.get("deep_solution", "B")
    work = Path(legacy["work_dir"]) if not blank(legacy.get("work_dir")) else db.parent / ("%s_legacy" % db.stem)
    work = resolve_path(work)
    work.mkdir(parents=True, exist_ok=True)
    master_rec, master_tag = master(db)
    if master_rec not in surface:
        raise ValueError("master_receiver %s must be one of surface_receivers (legacy needs its position known)" % master_rec)

    # Rerun safety: start every run from the imported receiver table and no derived tables.
    tables = set(query(db, "select name from sqlite_master where type = 'table'").name)
    if "tblReceiver_initial" in tables:
        execute(db, "drop table tblReceiver", "create table tblReceiver as select * from tblReceiver_initial")
    else:
        execute(db, "create table tblReceiver_initial as select * from tblReceiver")
    execute(db, *["drop table if exists %s" % t for t in DERIVED_TABLES + ["tblPositions_Deng"]])

    print("Phase 1: surface receivers %s, master %s (beacon %s)" % (surface, master_rec, master_tag))
    metronome(db, master_tag, work, method)
    clock_fix(db, surface, work)
    if deep:
        position_deep_receivers(db, surface, deep, work, method, solution)
        print("Phase 2: all receivers")
        execute(db, *["drop table if exists %s" % t for t in DERIVED_TABLES])
        metronome(db, master_tag, work, method)
        clock_fix(db, surface + deep, work)

    receivers = surface + deep
    tags = list(legacy.get("study_tags") or [])
    if not tags:
        tags = query(db, "select Tag_ID from tblTag where TagType = 'study' and pulseRate is not null").Tag_ID.tolist()
    missing_rate = query(db, "select Tag_ID from tblTag where TagType = 'study' and pulseRate is null").Tag_ID.tolist()
    if missing_rate:
        print("WARNING: study tags without pulseRate skipped (legacy epoch rule needs it): %s" % missing_rate)
        tags = [t for t in tags if t not in missing_rate]
    positions = Path(fresh_folder(work / "positions"))
    for tag in tags:
        print("Study tag %s" % tag)
        if tag_multipath(db, tag, work, method):
            deng(db, tag, receivers, positions, work / "figures")
    if any(positions.iterdir()):
        # positions_data_management adds solution and Tag_ID to every file
        widen_table(db, "tblPositions_Deng", positions, extra=("solution", "Tag_ID"))
        jsats3d.positions_data_management("Deng", str(positions), str(db))
    count = query(db, "select count(*) as n from sqlite_master where name = 'tblPositions_Deng'").n.iloc[0]
    solved = query(db, "select count(*) as n from tblPositions_Deng where comment = 'solution found'").n.iloc[0] if count else 0
    print("Done. tblPositions_Deng: %s solutions found. Work folder: %s" % (solved, work))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["import-2019", "process", "clock-fix-check"])
    parser.add_argument("run_file")
    parser.add_argument("receivers", nargs="*", help="Receivers for clock-fix-check")
    args = parser.parse_args(argv)
    run, paths = read_run_file(args.run_file)
    if args.command == "import-2019":
        import_2019(run, paths)
    elif args.command == "clock-fix-check":
        if not args.receivers:
            parser.error("clock-fix-check requires at least one receiver")
        clock_fix_check(paths["output_db"], args.receivers)
    else:
        process(run, paths)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as error:
        print("ERROR: %s" % error)
        sys.exit(2)
