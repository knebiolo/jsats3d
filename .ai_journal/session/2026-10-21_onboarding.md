# Session: Onboarding / Resume

- **Author:** AI assistant (OpenCode)
- **Timestamp:** 2026-10-21 17:55 local (PDT, UTC-7) / 2026-10-22 00:55 UTC
- **Tags:** onboarding, resume, staged-load

## Files touched
- Read (read-only): System_Prompt.txt, LONG_TERM_CONTEXT.md, .ai_journal/long_term/personality.md, README.md,
  .ai_journal/session/ (7 latest session files, all 2026-10-19), .gitignore, git status/log.
- Created: this journal file.
- No code, config, or data files modified.

## Short summary
Staged-load onboarding per System Prompt Section 12. Stage A complete: latest 6+ session files
(borderline 7th included, all from 2026-10-19), LONG_TERM_CONTEXT.md, personality.md, README.md.
Stage B not needed yet; nothing loaded beyond Stage A except .gitignore and git state.

### State as found (2026-10-19 → now)
- Repo at C:\Dev\jsat3d\jsats3d, branch `ENM_jsat3d_edits` (working branch, not main).
- Uncommitted on branch:
  - `jsats3d/jsats3d.py` — ZOI09 946 ms override + auto jump detection in `clock_fix()` (2026-10-19).
  - `scripts/ent_analysis_2025.py` — Kalman params 2.0/1.0/0.5/20.0 + empty-DataFrame guard (2026-10-19).
  - `LLM_Prompts.txt` modified; `scripts/metronome.py`, `scripts/mulitpath.py` deleted (cleanup, 2026-10-19).
  - New: `config/run_data_chinook_2025.toml`, `scripts/chinook_analysis_2025.py`.
  - 7 untracked 2026-10-19 journal files (see "Files touched").
- Pipeline status: ENT tag-drag (FC36/FFD3) DB `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db`
  rebuilt with clock fixes; 4 Chinook tags (0FBD, 041D, 493F, 5F82) run end-to-end
  (`output/jsats3d_2025_chinook.db`, plots + XYZ exports under `output/chinook_2025/`).
- Last session's key findings carried forward:
  - ZOI09 constant 946 ms offset fixed (→ 0.777 ms mean residual); ZOI02 jumps auto-detected.
  - Accuracy unchanged by clock fixes — dominant errors are receiver position uncertainty
    (CFD GPS 3–8 m std dev; ZOI08 beacon host unsurveyed) and flat array geometry
    (~9 m vertical spread, Y-axis scale distortion).
  - Dropping ZOI02 from `receivers_3d` recommended (3x Deng speedup, no accuracy loss) —
    NOT yet applied to any config file.
  - Receiver self-calibration not actionable (clock/position confounded); multi-beacon
    clock sync low ROI.
- Open owner escalations (LONG_TERM_CONTEXT.md, unchanged): ZOI02 reference-clock
  contradiction (2026-09-24), pairwise DBSCAN fixed parameters await owner review,
  ZOI05 UTC logging correction (applied in v2 DB rebuild).

## Decisions & assumptions
- Treated all 7 files dated 2026-10-19 as the "latest 6" window (7th is borderline;
  reading it costs nothing and it is the cleanup session that defines current script set).
- No new decisions taken. All prior decisions stand.

## Parameter changes with rationale
None this session.

## Blockers & known limitations
- Conda env not located: journal references `envs\jsat_3d\python.exe`; no `envs\` dir in
  repo root or workspace root, and `conda` is not on PATH in the current shell.
  Need env name/location confirmed before running any pipeline step.
- Uncommitted 2026-10-19 work (jsats3d.py clock fixes, chinook config/script, journal files)
  is not committed. No PR open. Gate 3 (multipath filter validation) and Gate 4
  (positioning acceptance) remain unreviewed by owner, so no PR per git workflow rules.

## Next steps
1. Confirm conda env name/location with user (blocks any pipeline run).
2. Decide with user: commit 2026-10-19 work to `ENM_jsat3d_edits`.
3. Decide with user: apply ZOI02 drop to `receivers_3d` (pending recommendation) or leave.
4. Await owner (Kevin) review of open escalations before further sync/multipath work.

## Active context
- Working dir: C:\Dev\jsat3d\jsats3d, branch ENM_jsat3d_edits.
- Raw data (READ-ONLY): K:\Jobs\5662\001\Data\DataTrans\2025_Data (25.4 GB, 487 files).
- Current DBs: output/jsats3d_2025_tagdrag_ent_anchorZOI03.db (ENT), output/jsats3d_2025_chinook.db (Chinook).
- Focus: awaiting user direction; no active implementation.
