# Session: Script Cleanup + Project Data Exploration

**Date:** 2026-10-19
**Goal:** Clean scripts directory to pipeline-only, then explore all project data on K drive

---

## Script Cleanup

Removed 8 diagnostic/discovery scripts that were never part of the pipeline:

**Removed:**
- `kalman_tune.py` — Kalman parameter tuning (standalone, 256 combos)
- `multi_beacon_clock.py` — Multi-beacon clock sync analysis (standalone)
- `debug_static_hold.py` — Debug script with hardcoded DB path
- `check_hold_times.py` — Debug script with hardcoded DB path
- `plot_fish_tracks.py` — Standalone plotting (not called by pipeline)
- `score_static_holds.py` — Standalone scoring (not called by pipeline)
- `metronome.py` — Standalone script with hardcoded paths (KPN author)
- `mulitpath.py` — Standalone multipath script (pipeline uses jsats3d module directly)

**Kept (7 pipeline scripts):**
- `tagdrag_2025_pipeline.py` — Main pipeline entry point
- `run_data.py` — Step 1 entry (parse raw → SQLite)
- `parse_ats_raw_to_legacy.py` — Data ingestion (imported by run_data)
- `adapt_2025_to_legacy.py` — Data adaptation (imported by parse_ats)
- `beacon_pairwise_dbscan.py` — DBSCAN beacon detection (imported)
- `legacy_pipeline.py` — Pipeline wrappers (imported)
- `ent_analysis_2025.py` — Step 12 analysis (subprocess)

Pipeline dependency chain:
```
run_data.py → parse_ats_raw_to_legacy.py → adapt_2025_to_legacy.py
run_data.py → beacon_pairwise_dbscan.py (subprocess)
run_data.py → legacy_pipeline.py (subprocess)
tagdrag_2025_pipeline.py → beacon_pairwise_dbscan (import)
tagdrag_2025_pipeline.py → legacy_pipeline (import)
tagdrag_2025_pipeline.py → ent_analysis_2025.py (subprocess)
```

---

## Project Data Exploration

User requested: read ALL project data from K drive to understand the project.
- Only read, never alter
- Focus on understanding data sources, structure, and relationships
