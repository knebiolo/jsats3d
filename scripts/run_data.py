"""Build a legacy jsats3d database from one filled-in run file, then optionally run the legacy workflow.

Usage (from the repo folder, env jsat_3d):
    python scripts/run_data.py                     reads config/run_data.toml (2025 ATS)
    python scripts/run_data.py config/run_data_2019.toml   2019 Teknologic (paper data)
    python scripts/run_data.py --dry-run           shows the plan, runs nothing
    python scripts/run_data.py --skip-build        reuse output_db; only apply [study] and run later steps

data_format = "ats"         2025 ATS raw files -> parse_ats_raw_to_legacy.py
data_format = "teknologic"  2019 Teknologic files -> shared legacy-compatible importer
Both give the same legacy tables. With [legacy] run = true, scripts/legacy_pipeline.py then runs
Kevin's preserved jsats3d.py workflow (metronome, clock fix, deep receivers, study tags, Deng)
in the shared environment.

Species are turned into acoustic tag codes through PTAGIS released_v0 ("Acoustic Tag Value").
Beacons are always added when tags are filtered, so clock synchronization stays possible.
Raw data is read only: the run refuses to write inside an input folder or to replace an
existing database unless overwrite = true.
"""
import argparse
import itertools
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from parse_ats_raw_to_legacy import load_target_receivers  # noqa: E402

DEFAULT_RUN_FILE = REPO / "config" / "run_data.toml"
PARSER = REPO / "scripts" / "parse_ats_raw_to_legacy.py"
DBSCAN = REPO / "scripts" / "beacon_pairwise_dbscan.py"
LEGACY = REPO / "scripts" / "legacy_pipeline.py"

FORMATS = {
    "ats": (["config_xlsx", "gps_csv", "covariate_csv", "temperature_csv", "released_file"], ["raw_root", "hobo_dir"]),
    "teknologic": (["tag_csv", "receiver_csv", "wsel_csv", "temp_csv"], ["raw_root"]),
}
INPUT_FILES, INPUT_DIRS = FORMATS["ats"]
STUDY_COLUMNS = ["UTC_Conv", "BM_Elev", "BM_Elev_Units", "Output_Units", "masterReceiver",
                 "synch_time_start", "synch_time_end"]
HEX4 = re.compile(r"^[0-9A-F]{4}$")
SCIENTIFIC = re.compile(r"^\d+(\.\d+)?E[+-]?\d+$")
# Every 4-character code of the form digits + "E" + digits (e.g. 41E0, 2E39). Excel reads these as
# numbers and shows them in scientific notation (41E0 -> 4.10E+01), which is how released_v0 was damaged.
EXPONENT_CODES = [
    "".join(m) + "E" + "".join(e)
    for size in (1, 2)
    for m in itertools.product("0123456789", repeat=size)
    for e in itertools.product("0123456789", repeat=3 - size)
]


def normalize_acoustic_code(value):
    """Return (4-hex code, repair note) or (None, reason).

    Excel damage is undone only when exactly one 4-character code gives the same number;
    anything else is reported, never guessed.
    """
    if value is None or pd.isna(value) or not str(value).strip():
        return None, "blank"
    text = str(value).strip().upper()
    if HEX4.match(text):
        return text, None
    if text.isdigit() and len(text) < 4:
        return text.zfill(4), "leading zeros restored"
    if SCIENTIFIC.match(text):
        target = float(text)
        matches = [code for code in EXPONENT_CODES if math.isclose(float(code), target, rel_tol=1e-9, abs_tol=0.0)]
        if len(matches) == 1:
            return matches[0], "scientific notation reversed"
        return None, "ambiguous scientific notation (%s candidates)" % len(matches)
    return None, "not a 4-character hex code"


def read_table(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        table = pd.read_excel(path, dtype=str)
    else:
        table = pd.read_csv(path, dtype=str)
    table.columns = table.columns.astype(str).str.strip()
    return table


def load_species_tags(released_file, species):
    """Acoustic codes for the requested species. Returns (codes, repaired, unresolved)."""
    released = read_table(released_file)
    for column in ("Acoustic Tag Value", "Species Name"):
        if column not in released.columns:
            raise ValueError("%s has no '%s' column; columns are %s" % (released_file, column, list(released.columns)))
    acoustic = released[released["Acoustic Tag Value"].fillna("").str.strip() != ""].copy()
    acoustic["species_key"] = acoustic["Species Name"].fillna("").str.strip().str.lower()
    available = sorted(acoustic["Species Name"].dropna().str.strip().unique())
    wanted = {name.strip().lower() for name in species}
    unknown = sorted(wanted - set(acoustic["species_key"]))
    if unknown:
        raise ValueError("Species %s have no acoustic tags in %s. Species with acoustic tags: %s"
                         % (unknown, released_file, available))
    codes, repaired, unresolved = set(), [], []
    for raw in acoustic.loc[acoustic["species_key"].isin(wanted), "Acoustic Tag Value"]:
        code, note = normalize_acoustic_code(raw)
        if code is None:
            unresolved.append((raw, note))
        else:
            codes.add(code)
            if note:
                repaired.append((raw, code))
    return codes, repaired, unresolved


def normalize_tags(tags):
    cleaned = {str(tag).strip().upper() for tag in tags if str(tag).strip()}
    bad = sorted(tag for tag in cleaned if not HEX4.match(tag))
    if bad:
        raise ValueError("Tags must be 4 hex characters (e.g. FFD3); got %s" % bad)
    return cleaned


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path



def read_run_file(path):
    with open(path, "rb") as stream:
        run = tomllib.load(stream)
    run["paths"] = {name: resolve_path(value) for name, value in run.get("paths", {}).items() if str(value).strip()}
    for section in ("selection", "study", "legacy", "dbscan"):
        run.setdefault(section, {})
    run.setdefault("data_format", "ats")
    if run["data_format"] not in FORMATS:
        raise ValueError("data_format must be one of %s, got %r" % (sorted(FORMATS), run["data_format"]))
    return run


def check_inputs(paths, data_format="ats"):
    files, dirs = FORMATS[data_format]
    problems = []
    for name in files + dirs + ["output_db"]:
        if name not in paths:
            problems.append("%s: not filled in" % name)
    for name in files:
        if name in paths and not paths[name].is_file():
            problems.append("%s: file not found: %s" % (name, paths[name]))
    for name in dirs:
        if name in paths and not paths[name].is_dir():
            problems.append("%s: folder not found: %s" % (name, paths[name]))
    if problems:
        raise ValueError("Fix the run file:\n  " + "\n  ".join(problems))


def check_output(output_db, paths, overwrite, data_format="ats"):
    """Refuse to write into the raw data tree or silently replace a database."""
    files, dirs = FORMATS[data_format]
    bases = [paths[name] if name in dirs else paths[name].parent for name in files + dirs]
    bases = [base.resolve() for base in bases]
    common = Path(os.path.commonpath([str(base) for base in bases]))
    if common.parent != common:  # skip when the only shared folder is a drive root
        bases.append(common)
    target = output_db.resolve()
    for base in bases:
        if target.is_relative_to(base):
            raise ValueError("output_db %s is inside input folder %s (raw data is read only)" % (output_db, base))
    if output_db.exists() and not overwrite:
        raise ValueError("output_db %s already exists. Pick a new name, set overwrite = true, "
                         "or use --skip-build to reuse it." % output_db)


def check_time(selection):
    start, end = selection.get("start") or None, selection.get("end") or None
    parsed = [datetime.fromisoformat(str(value)) if value else None for value in (start, end)]
    if parsed[0] and parsed[1] and parsed[0] >= parsed[1]:
        raise ValueError("start %s must be before end %s" % (start, end))
    return start and str(start), end and str(end)


def receiver_serials(config_xlsx, receivers):
    if not receivers:
        return []
    targets = load_target_receivers(config_xlsx)
    by_name = {name: serial for serial, name in targets["Receiver Name"].items()}
    unknown = sorted(set(receivers) - set(by_name))
    if unknown:
        raise ValueError("Unknown receivers %s. Valid: %s" % (unknown, sorted(by_name)))
    return [by_name[name] for name in receivers]


def parser_command(paths, start, end, serials, tags):
    command = [
        sys.executable, str(PARSER), str(paths["raw_root"]),
        "--config-xlsx", str(paths["config_xlsx"]),
        "--output-db", str(paths["output_db"]),
        "--legacy-db",
        "--temperature-csv", str(paths["temperature_csv"]),
        "--hobo-dir", str(paths["hobo_dir"]),
        "--gps-csv", str(paths["gps_csv"]),
        "--covariate-csv", str(paths["covariate_csv"]),
    ]
    if start:
        command += ["--start", start]
    if end:
        command += ["--end", end]
    for serial in serials:
        command += ["--serial", serial]
    if tags:
        for tag in sorted(tags):
            command += ["--tag", tag]
        # Tag filter on: add every configured beacon so clock sync stays possible.
        command.append("--include-config-beacons")
    return command


def legacy_command(legacy, subcommand, run_file):
    """Run the preserved legacy workflow in the active shared environment."""
    return [sys.executable, str(LEGACY), subcommand, str(run_file)]


def dbscan_command(output_db, dbscan, start, end):
    begin, finish = dbscan.get("start") or start, dbscan.get("end") or end
    if not (begin and finish):
        raise ValueError("[dbscan] needs start and end (set them in [dbscan] or [selection])")
    output_dir = resolve_path(dbscan.get("output_dir") or "output/dbscan_%s" % output_db.stem)
    return [
        sys.executable, str(DBSCAN), str(output_db),
        "--beacon-receiver", dbscan.get("beacon_receiver", "ZOI02"),
        "--anchor", dbscan.get("anchor", "ZOI09"),
        "--start", str(begin), "--end", str(finish),
        "--output-dir", str(output_dir),
    ]


def study_row(study, start, end):
    """tblStudyParameters values from [study]; the sync window defaults to the selection time frame."""
    def value(key, default=None):
        item = study.get(key, default)
        return None if isinstance(item, str) and not item.strip() else item
    return [value("utc_conv"), value("bm_elev"), value("bm_elev_units", "feet"), value("output_units", "meters"),
            value("master_receiver"), value("synch_time_start") or start, value("synch_time_end") or end]


def finish_database(db, study, start, end, signal_proxies, pulse_rates=None):
    """Fill tblStudyParameters, pulse rates from the run file, optional ATS stand-ins for SNR/NBW, and indexes."""
    connection = sqlite3.connect(db)
    try:
        connection.execute("DROP TABLE IF EXISTS tblStudyParameters")
        connection.execute("CREATE TABLE tblStudyParameters(UTC_Conv INTEGER, BM_Elev REAL, BM_Elev_Units TEXT, "
                           "Output_Units TEXT, masterReceiver TEXT, synch_time_start TIMESTAMP, synch_time_end TIMESTAMP)")
        connection.execute("INSERT INTO tblStudyParameters VALUES (?,?,?,?,?,?,?)", study_row(study, start, end))
        for tag, rate in (pulse_rates or {}).items():
            updated = connection.execute("UPDATE tblTag SET pulseRate = ? WHERE Tag_ID = ?", (float(rate), tag.upper())).rowcount
            print("pulseRate %s = %s s%s" % (tag.upper(), rate, "" if updated else " (tag not in tblTag, ignored)"))
        filled = 0
        if signal_proxies:
            # PROVISIONAL (Kevin to approve): ATS has no SNR/NBW. Legacy multipath_classifier keeps SNR > 0 and
            # trains on Amplitude, NBW, SNR. Amplitude is already SigStr.
            filled = connection.execute(
                "UPDATE tblDetectionRaw SET SNR = SigStr - Threshold, "
                "NBW = CAST(CASE WHEN instr(trim(BitPeriod), ' ') > 0 "
                "THEN substr(trim(BitPeriod), 1, instr(trim(BitPeriod), ' ') - 1) ELSE trim(BitPeriod) END AS REAL) "
                "WHERE SNR IS NULL AND SigStr IS NOT NULL AND Threshold IS NOT NULL").rowcount
        # Legacy queries every tag and receiver with WHERE Tag_ID / Rec_ID; indexes change speed, not results.
        raw_columns = {row[1] for row in connection.execute("PRAGMA table_info(tblDetectionRaw)")}
        indexes = [("idx_raw_tag_rec", ("Tag_ID", "Rec_ID")), ("idx_raw_rec", ("Rec_ID",)),
                   ("idx_raw_tag_rec_seconds", ("Tag_ID", "Rec_ID", "seconds")),
                   ("idx_raw_tag_seconds", ("Tag_ID", "seconds"))]
        for name, columns in indexes:
            if set(columns) <= raw_columns:
                connection.execute("CREATE INDEX IF NOT EXISTS %s ON tblDetectionRaw (%s)" %
                                   (name, ", ".join(columns)))
        connection.commit()
    finally:
        connection.close()
    print("tblStudyParameters: %s" % dict(zip(STUDY_COLUMNS, study_row(study, start, end))))
    if signal_proxies:
        print("PROVISIONAL: %s rows given SNR = SigStr - Threshold, NBW = BitPeriod (pending Kevin)" % filled)
    return filled


def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "uncommitted_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_file", nargs="?", default=str(DEFAULT_RUN_FILE))
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and print the plan without running")
    parser.add_argument("--skip-build", action="store_true", help="Reuse an existing output_db")
    return parser.parse_args(argv)


def prepare_run(args):
    run_file = Path(args.run_file).resolve()
    run = read_run_file(run_file)
    paths, selection, study, legacy, dbscan = (run[k] for k in ("paths", "selection", "study", "legacy", "dbscan"))
    data_format = run["data_format"]
    check_inputs(paths, data_format)
    output_db = paths["output_db"]
    if args.skip_build:
        if not output_db.is_file():
            raise ValueError("--skip-build: output_db %s does not exist" % output_db)
    else:
        check_output(output_db, paths, bool(run.get("overwrite", False)), data_format)
    start, end = check_time(selection)
    return args, run_file, run, paths, selection, study, legacy, dbscan, output_db, start, end


def select_inputs(data_format, paths, selection):
    if data_format != "ats":
        return set(), [], [], set(), []
    serials = receiver_serials(paths["config_xlsx"], selection.get("receivers") or [])
    species = selection.get("species") or []
    if not species:
        return set(), [], [], normalize_tags(selection.get("tags") or []), serials
    codes, repaired, unresolved = load_species_tags(paths["released_file"], species)
    return codes, repaired, unresolved, codes | normalize_tags(selection.get("tags") or []), serials


def print_run_summary(run_file, output_db, data_format, start, end, selection, tags,
                      species_codes, repaired, unresolved, study, legacy):
    print("Run file:    %s" % run_file)
    print("Data format: %s" % data_format)
    print("Output DB:   %s" % output_db)
    if data_format == "ats":
        print("Time (PDT):  %s to %s" % (start or "first detection", end or "last detection"))
        print("Receivers:   %s" % (", ".join(selection.get("receivers")) if selection.get("receivers") else "all 20 targets"))
        if species_codes:
            print("Species:     %s -> %s acoustic tags" % (", ".join(selection["species"]), len(species_codes)))
        print("Tags:        %s" % ("%s study tags + all configured beacons" % len(tags) if tags else "ALL tags (no filter)"))
        for raw, code in repaired:
            print("REPAIRED PTAGIS code %r -> %s (Excel damage; ask PM for a text re-export)" % (raw, code))
        for raw, reason in unresolved:
            print("WARNING PTAGIS code %r skipped: %s" % (raw, reason))
    else:
        print("Teknologic import takes every tag inside the [study] sync window (legacy behaviour); choose tags to position with [legacy] study_tags.")
    if legacy.get("run"):
        print("Legacy:      master %s, surface %s, deep %s, method %s, env %s"
              % (study.get("master_receiver"), legacy.get("surface_receivers"), legacy.get("deep_receivers"),
                 legacy.get("method", "KNN"), legacy.get("env") or "jsat_legacy"))


def build_steps(args, run_file, data_format, paths, selection, study, legacy, dbscan, output_db, start, end, tags):
    ats = data_format == "ats"
    steps = []
    if not args.skip_build:
        if ats:
            serials = receiver_serials(paths["config_xlsx"], selection.get("receivers") or [])
            steps.append(("parse ATS raw files", parser_command(paths, start, end, serials, tags)))
        else:
            steps.append(("legacy 2019 import", legacy_command(legacy, "import-2019", run_file)))
    proxies = ats and bool(legacy.get("signal_proxies", False))
    rates = study.get("pulse_rates") or {}
    steps.append(("study parameters and indexes", lambda: finish_database(output_db, study, start, end, proxies, rates)))
    if legacy.get("run"):
        missing = [key for key in ("bm_elev", "master_receiver") if str(study.get(key, "")).strip() == ""]
        if missing:
            raise ValueError("[legacy] run needs [study] %s: legacy clock_fix_object reads BM_Elev "
                             "(converts it by units) and masterReceiver" % " and ".join(missing))
        steps.append(("legacy workflow", legacy_command(legacy, "process", run_file)))
    if dbscan.get("run"):
        if not ats:
            raise ValueError("[dbscan] runs on ATS data only")
        needed = {dbscan.get("beacon_receiver", "ZOI02"), dbscan.get("anchor", "ZOI09")}
        receivers = set(selection.get("receivers") or [])
        if receivers and not needed <= receivers:
            raise ValueError("[dbscan] needs receivers %s in [selection] receivers" % sorted(needed - receivers))
        steps.append(("pairwise beacon DBSCAN", dbscan_command(output_db, dbscan, start, end)))
    return steps


def run_steps(steps, manifest, manifest_path):
    for name, step in steps:
        print("\n=== %s ===" % name)
        sys.stdout.flush()
        began = datetime.now()
        if callable(step):
            step()
            code, command = 0, None
        else:
            code, command = subprocess.run(step, cwd=REPO).returncode, step
        manifest["steps"].append({"step": name, "command": command, "return_code": code,
                                  "minutes": round((datetime.now() - began).total_seconds() / 60, 1)})
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        if code != 0:
            print("STOPPED: %s failed (exit %s). Run record: %s" % (name, code, manifest_path))
            return code
    return 0


def main(argv=None):
    args, run_file, run, paths, selection, study, legacy, dbscan, output_db, start, end = prepare_run(parse_args(argv))
    data_format = run["data_format"]
    species_codes, repaired, unresolved, tags, serials = select_inputs(data_format, paths, selection)
    print_run_summary(run_file, output_db, data_format, start, end, selection, tags,
                      species_codes, repaired, unresolved, study, legacy)
    steps = build_steps(args, run_file, data_format, paths, selection, study, legacy, dbscan, output_db, start, end, tags)
    if args.dry_run:
        print("Dry run: nothing run. Steps: %s" % "; ".join(name for name, _ in steps))
        return 0

    manifest = {
        "run_file": str(run_file),
        "started": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "data_format": data_format,
        "paths": {name: str(path) for name, path in paths.items()},
        "selection": selection, "study": study, "legacy": legacy, "dbscan": dbscan,
        "tags": sorted(tags),
        "ptagis_repaired": [[str(raw), code] for raw, code in repaired],
        "ptagis_unresolved": [[str(raw), reason] for raw, reason in unresolved],
        "steps": [],
    }
    manifest_path = output_db.with_suffix(".run.json")
    output_db.parent.mkdir(parents=True, exist_ok=True)
    code = run_steps(steps, manifest, manifest_path)
    if code:
        return code
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print("\nDone. Database: %s\nRun record: %s" % (output_db, manifest_path))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as error:
        print("ERROR: %s" % error)
        sys.exit(2)
