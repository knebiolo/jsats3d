"""Build a legacy jsats3d database from one filled-in run file.

Usage (from the repo folder, env jsat_3d):
    python scripts/run_data.py                  reads config/run_data.toml
    python scripts/run_data.py my_run.toml      reads another run file
    python scripts/run_data.py --dry-run        shows the plan, parses nothing

Species are turned into acoustic tag codes through PTAGIS released_v0 ("Acoustic Tag Value").
Beacons are always added when tags are filtered, so clock synchronization stays possible.
Raw data is read only: the run refuses to write inside an input folder or to replace an
existing database unless overwrite = true. The work itself is done by
parse_ats_raw_to_legacy.py (and optionally beacon_pairwise_dbscan.py); nothing is re-implemented.
"""
import argparse
import itertools
import json
import math
import os
import re
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

INPUT_FILES = ["config_xlsx", "gps_csv", "covariate_csv", "temperature_csv", "released_file"]
INPUT_DIRS = ["raw_root", "hobo_dir"]
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
    paths = {name: resolve_path(value) for name, value in run.get("paths", {}).items() if str(value).strip()}
    selection = run.get("selection", {})
    return paths, selection, run.get("dbscan", {}), bool(run.get("overwrite", False))


def check_inputs(paths):
    problems = []
    for name in INPUT_FILES + INPUT_DIRS + ["output_db"]:
        if name not in paths:
            problems.append("%s: not filled in" % name)
    for name in INPUT_FILES:
        if name in paths and not paths[name].is_file():
            problems.append("%s: file not found: %s" % (name, paths[name]))
    for name in INPUT_DIRS:
        if name in paths and not paths[name].is_dir():
            problems.append("%s: folder not found: %s" % (name, paths[name]))
    if problems:
        raise ValueError("Fix the run file:\n  " + "\n  ".join(problems))


def check_output(output_db, paths, overwrite):
    """Refuse to write into the raw data tree or silently replace a database."""
    bases = [paths[name] if name in INPUT_DIRS else paths[name].parent for name in INPUT_FILES + INPUT_DIRS]
    bases = [base.resolve() for base in bases]
    common = Path(os.path.commonpath([str(base) for base in bases]))
    if common.parent != common:  # skip when the only shared folder is a drive root
        bases.append(common)
    target = output_db.resolve()
    for base in bases:
        if target.is_relative_to(base):
            raise ValueError("output_db %s is inside input folder %s (raw data is read only)" % (output_db, base))
    if output_db.exists() and not overwrite:
        raise ValueError("output_db %s already exists. Pick a new name or set overwrite = true." % output_db)


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
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and print the plan without parsing")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_file = Path(args.run_file).resolve()
    paths, selection, dbscan, overwrite = read_run_file(run_file)
    check_inputs(paths)
    output_db = paths["output_db"]
    check_output(output_db, paths, overwrite)
    start, end = check_time(selection)
    serials = receiver_serials(paths["config_xlsx"], selection.get("receivers") or [])

    species = selection.get("species") or []
    species_codes, repaired, unresolved = set(), [], []
    if species:
        species_codes, repaired, unresolved = load_species_tags(paths["released_file"], species)
    extra_tags = normalize_tags(selection.get("tags") or [])
    tags = species_codes | extra_tags

    print("Run file:   %s" % run_file)
    print("Output DB:  %s" % output_db)
    print("Time (PDT): %s to %s" % (start or "first detection", end or "last detection"))
    print("Receivers:  %s" % (", ".join(selection.get("receivers")) if serials else "all 20 targets"))
    if species:
        print("Species:    %s -> %s acoustic tags" % (", ".join(species), len(species_codes)))
    if extra_tags:
        print("Extra tags: %s" % ", ".join(sorted(extra_tags)))
    print("Tags:       %s" % ("%s study tags + all configured beacons" % len(tags) if tags else "ALL tags (no filter)"))
    for raw, code in repaired:
        print("REPAIRED PTAGIS code %r -> %s (Excel damage; ask PM for a text re-export)" % (raw, code))
    for raw, reason in unresolved:
        print("WARNING PTAGIS code %r skipped: %s" % (raw, reason))

    commands = [parser_command(paths, start, end, serials, tags)]
    if dbscan.get("run"):
        needed = {dbscan.get("beacon_receiver", "ZOI02"), dbscan.get("anchor", "ZOI09")}
        receivers = set(selection.get("receivers") or [])
        if receivers and not needed <= receivers:
            raise ValueError("[dbscan] needs receivers %s in [selection] receivers" % sorted(needed - receivers))
        commands.append(dbscan_command(output_db, dbscan, start, end))
    if args.dry_run:
        print("Dry run: nothing parsed. Steps: %s" % ", ".join(Path(command[1]).name for command in commands))
        return 0

    manifest = {
        "run_file": str(run_file),
        "started": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "paths": {name: str(path) for name, path in paths.items()},
        "selection": selection,
        "dbscan": dbscan,
        "tags": sorted(tags),
        "ptagis_repaired": [[str(raw), code] for raw, code in repaired],
        "ptagis_unresolved": [[str(raw), reason] for raw, reason in unresolved],
        "steps": [],
    }
    manifest_path = output_db.with_suffix(".run.json")
    output_db.parent.mkdir(parents=True, exist_ok=True)
    for command in commands:
        step = Path(command[1]).name
        print("\n=== %s ===" % step)
        sys.stdout.flush()
        began = datetime.now()
        code = subprocess.run(command, cwd=REPO).returncode
        manifest["steps"].append({"step": step, "command": command, "return_code": code,
                                  "minutes": round((datetime.now() - began).total_seconds() / 60, 1)})
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        if code != 0:
            print("STOPPED: %s failed (exit %s). Run record: %s" % (step, code, manifest_path))
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
