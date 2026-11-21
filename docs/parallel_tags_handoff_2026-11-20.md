# Parallel per-tag processing (positions + fish_multipath): implementation and test instructions (2026-11-20)

For: Qwen 3.6 35B MoE session. Read `LONG_TERM_CONTEXT.md` first. Conda env `jsat_3d` (Windows). Work from the
repo root. Line numbers refer to commit `b20a615` (branch `ENM_jsats3d_Updated`).

## Goal

Process many fish tags at the same time in separate worker processes, in the two slowest steps that run once per fish:

| Step (100-fish run, 2026-11-19) | Minutes | Function |
|---|---|---|
| positions (Deng 3D) | 388 | `positions_run`, `scripts/_2025_pipeline.py:479` |
| fish_multipath | 116 | `fish_multipath_run`, `scripts/_2025_pipeline.py:410` |

Both loop `for tag in c.tags:` in one process. Each tag's math does not depend on any other tag, so the work can run in
parallel. Target: 600 tags. The math is not changed.

## Hard constraints

- **HC-1, identical results.** Every output must be byte-identical to the serial code: the same CSV bytes, the same
  table rows, the same row order in `tblDetectionFilterPrimary` / `tblDetectionFilterSecondary`, and the same
  `notes` dict contents and key order in the status JSON. Console lines from different workers may interleave. That
  is the only allowed difference.
- **HC-2, no change to `jsats3d/jsats3d.py`.** Edit only `scripts/_2025_pipeline.py`, plus the new test files listed
  below. Do not edit `legacy_pipeline.py`, the run files of finished runs, or any database except the test copy.
- **HC-3, workers = 1 is today's code path.** With one worker, the step runs the existing in-process loop exactly as
  now (no pool). Parallelism changes only where the work runs, never the math.
- **No git push.** Commit only if the user asks. The user pushes.

## Design

### Setting

- New CLI flag `--workers N` in `main()` (`_2025_pipeline.py:1001`), and an optional run-file key
  `[performance] workers = N`. The CLI wins over the run file. Default 1. Store it as `c.workers` in `build_context`
  (`_2025_pipeline.py:915`).
- Add the worker count to the startup print line.

### Windows rules (spawn start method)

- Worker functions must be **module-level** functions in `_2025_pipeline.py` that take only plain picklable args
  (`str` paths, `str` tag, `list[str]` receivers). Do not pass `c` or DataFrames.
- `main()` is already behind `if __name__ == "__main__":`. Keep it that way.
- Use `multiprocessing.get_context("spawn").Pool(processes=N, maxtasksperchild=1)` with `apply_async`.
  `maxtasksperchild=1` gives each tag a fresh process, so pandas memory is released after every tag.
- Workers return **small dicts only** (counts, timing, peak memory). Never return the solution frames (millions of
  string rows). Pickling them would cost more time and memory than parallelism saves.
- On any worker exception: `pool.terminate()`, then fail the step with the tag name in the message.

### positions (`positions_run`)

Worker `_positions_worker(db, tag, receivers_3d, out, figures)`:
```python
pos = lp.deng(Path(db), tag, receivers_3d, Path(out), Path(figures))
if pos is None:
    return {"tag": tag, "none": True}
return {"tag": tag, "none": False,
        "A_found": int((pos.DengSolutionA_unfiltered.comment == "solution found").sum()),
        "B_found": int((pos.DengSolutionB_unfiltered.comment == "solution found").sum())}
```
- Call `out = lp.fresh_folder(c.work / "positions")` once, before submitting anything (as now). Deng writes
  `<TAG>_solutionA/B.csv`, a different file name per tag, so workers never write the same file.
- Submit tags **largest first**, so the biggest tag does not start last and leave one worker running alone at the
  end. Size = `select Tag_ID, count(*) from tblDetectionFilterSecondary group by Tag_ID` (one query before
  submitting).
- Build `notes` **in `c.tags` order** after all results are in. If any result has `none: True`, raise the same
  `RuntimeError("check failed: no filtered detections for %s" % tag)` for the **first such tag in `c.tags` order**.
  Today the step stops at the first empty tag. In parallel, the other tags finish first and leave CSVs behind, but the
  step is marked FAILED either way and a rerun redoes it, so this is acceptable.
- Keep the final `if not any(out.glob("*.csv"))` check.
- This step does not write the database, so parallel readers are safe.

### fish_multipath (`fish_multipath_run`): compute in parallel, write serially

This step writes the database. SQLite allows one writer, and table row order must match the serial run, so split
each tag into a compute part and a write part.

- **Compute (parallel worker)** `_multipath_worker(db, tag, folder)`: run `jsats3d.multipath_data_object(tag, db,
  folder)`. If `data.empty`, return `{"tag": tag, "empty": True}`. Otherwise run `jsats3d.multipath_2(data)` (writes
  `<TAG>_<REC>_multipath.csv` into `folder`) and return `{"tag": tag, "empty": False}`. Each tag gets its own folder,
  `c.work / "multipath_primary" / tag`. Fresh-folder the parent once at the start. The compute part reads only
  `tblTag`, `tblStudyParameters`, `tblDetectionRaw` and the clock tables. This step writes none of them, so a tag's
  result cannot depend on other tags' writes.
- **Write (main process, strictly in `c.tags` order)**: the current loop body from `existing = table_names(c.db)`
  through the `notes[tag] = ...` line, unchanged except that `primary_folder` is the tag's own folder. If a tag's
  compute result is `empty`, raise the same `RuntimeError("no clock-fixed detections for study tag %s" % tag)` at
  that point in the order. Earlier tags are already written, exactly as now.
- **Batches.** Do not compute all 600 tags before writing. Process `c.tags` in batches of `4 * workers` tags. Run the
  batch's compute in parallel, then write that batch serially, then start the next batch. Workers and the writer
  never run at the same time, so there is no "database is locked" risk. Disk use stays bounded too, because
  `multipath_data_management` deletes each CSV after loading it.
- `lp.ensure_progress`, the `__ats_fish_two_stage__` delete at the start, and `lp.mark_done` at the end stay where
  they are.
- `os.listdir` order inside a tag's folder is the same as today (same file names, and the folder holds only that
  tag), so `widen_table` and `multipath_data_management` load files in the same order.

## Phase 0: measure before writing code (report the numbers to the user)

Use the current serial code on 5 tags, including the largest:

1. positions: per-tag wall time and **peak RSS** of `lp.deng(...)` writing into a scratch folder. Sample with
   `psutil.Process().memory_info().rss` from a background thread.
2. fish_multipath: split each tag's time into **compute** (`multipath_data_object` + `multipath_2`) and **write**
   (delete + `widen_table` + `multipath_data_management` + read-back + `classify_ats_fish` + `to_sql`). Do this on the
   DB copy (see Phase 2), because the write part modifies the DB. If the write part dominates, parallel compute will
   not help much. **Report this split before implementing the fish_multipath part.**
3. Report CPU cores (`os.cpu_count()`), total and free RAM, and free disk on the output drive.
4. Run this read-only query: `select name, tbl_name, sql from sqlite_master where type = 'index'`. Report whether
   `tblDetectionFilterSecondary` and `tblDetectionFilterPrimary` have an index starting with `Tag_ID`. No code in the
   repo creates one. Deng's `position.__init__` queries Secondary once per receiver per tag (20 queries per tag), and
   fish_multipath deletes and reads back by `Tag_ID`. Without an index, every one of those is a full table scan of a
   ~50 GB database. **Report only. Do not add an index.** Adding one is the user's decision.

Choose the default worker count from the measurements as
`min(os.cpu_count() - 1, floor((free_RAM_GB - 4) / peak_RSS_GB))`. Tell the user the number and how you got it.

## Phase 1: implement positions, then test

positions does not write the database. Test it against the real DB, writing only into scratch output folders.

### Equivalence script (`scripts/_perf_parallel_check.py`, delete at sign-off)

1. Pick 6 tags from the run file, including the largest by Secondary row count and one with few rows.
2. Build a context the same way `build_context` builds `c`, but with `work` pointing at a scratch folder and `tags`
   set to the 6. Run `positions_run` with `workers=1` into scratch folder A, then with `workers=4` into scratch
   folder B.
3. Pass criteria: the same set of file names; **SHA-256 identical** for every `*_solution?.csv`; `notes` equal
   including key order (`list(notes.items())`).
4. Report wall time for workers=1 vs workers=4.

### Unit test (`tests/test_parallel_tags.py`, keep)

Factor the ordering and error logic into small plain functions (for example "given results keyed by tag and
`c.tags`, build `notes` in order and find the first empty tag"). Unit-test those functions directly: notes order
equals `c.tags` order whatever the completion order, and the first empty tag in `c.tags` order raises the same
`RuntimeError` text as the serial code. Do not try to monkeypatch into spawned processes. The equivalence script
covers the real pool.

Existing tests must still pass: `python -m pytest tests -q`.

## Phase 2: implement fish_multipath, then test

Do this only if Phase 0 item 2 shows that compute is a meaningful share of the step.

### Test database

fish_multipath writes, so use a **copy** of the DB (`--db` points the pipeline at the copy, which gets its own work
folder). The DB is about 50 GB. Check free disk first. If there is not room for the copy plus 20 GB, **stop and ask
the user** where to put it. Never write to the production DB. Never use the K: drive as a target.

### Equivalence test

1. Make a scratch copy of the run file that changes only `[selection] tags` (the same 6 tags) and `output_db`.
2. Serial run on the copy: `python scripts/_2025_pipeline.py <scratch run file> --only fish_multipath --workers 1`.
   Then dump `select * from tblDetectionFilterPrimary where Tag_ID in (...) order by rowid` to CSV, do the same for
   Secondary, and SHA-256 both dumps.
3. The step skips itself when `fish_multipath_done` (`_2025_pipeline.py:383`) says it is done. Read that function and
   clear whatever it checks **on the copy only**, so the step runs again.
4. Parallel run on the same copy: `--only fish_multipath --workers 4`. The step deletes each tag's rows and
   re-appends them, so rowids change but the relative order must not. Dump again with the same queries.
5. Pass criteria: SHA-256 identical for both dumps, identical `notes`, and no "database is locked" in the log.

## Phase 3: one end-to-end timing run (the user decides when)

Ask the user before launching. Suggest 20 tags with `--from fish_multipath --to export` on a DB copy. Report
minutes per step at workers=1 (from the 11-19 status JSON, scaled per tag) vs workers=N, and peak RAM.

## Report back to the user

- Phase 0 numbers: per-tag time and peak RSS, the compute/write split, cores, RAM, disk, the index check result.
- Files changed, with one line on each.
- Equivalence results: hashes match yes/no, how many files and rows were compared.
- Measured speedup, and the projected time for 600 tags at the chosen worker count.
- Anything that did not match, with the exact difference. Do not "fix" a difference by changing the math.

## Out of scope

- Vectorising Deng's per-combination loop ("Stage B"). It changes last-digit values and needs a tolerance from the
  project lead.
- Adding database indexes, changing the SQLite journal mode (WAL), or editing `jsats3d.py`.
- Parallelising other steps (export, DBSCAN, figures). Say in the report if they become the slowest steps.
