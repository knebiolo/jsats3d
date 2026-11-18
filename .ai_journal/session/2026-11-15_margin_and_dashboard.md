# 3D Box Margin Decision + fish_dashboard_v3 Port — Session Journal
# Session: 2026-11-15
# Branch: ENM_jsats3d_clock_updates (only branch touched)

## Tasks (owner)
1. Run the full 4-tag fish_figures pass (box on + presence, from the run file)
   and check fish_qc.csv for tags showing clipped near-edge fixes — if so, that
   is the evidence for adding a margin.
2. Bring scripts/fish_dashboard_v3.py over to this branch (it exists only on
   the frozen ENM_jsats3d_edits_clocksync branch).

## Task 1 — 4-tag pass, margin=0, then margin evidence, then margin=10

Production DB on disk: output/10292026_Test/jsats3d_2025_earlyfish.db (Arm A
unshifted; the original output/jsats3d_2025_earlyfish.db no longer exists on
disk). Run:
  python scripts/fish_figures_2025.py --db output/10292026_Test/jsats3d_2025_earlyfish.db \
      --out output/pm_figures_2026-11-15/example_fish --run-file config/run_data_earlyfish_2025.toml
Margin=0 pass logged to output/11152026_4tag_pass.txt (kept; the .log
PowerShell redirect was UTF-16 and read as binary, so an ASCII .txt copy was
made and the .log deleted).

Margin evidence (full-study per-tag analysis, throwaway script deleted,
outputs kept in output/11152026_4tag_margin/: margin_analysis.csv + one
< TAG>_dropped_fixes.csv per tag with per-fix Chebyshev distance to the
margin-0 box, negative = still inside):

  tag   dropped  inside-box  dist_p50  dist_p90  dist_max  <=5m  <=10m  <=25m  >50m
  26E4    127        6        3.1 m    12.7 m    37.0 m    79    103    116     0
  9F8C     34        0       12.8 m    24.7 m    33.5 m     6     16     31     0
  1D5A    730       58        5.1 m    10.8 m    32.7 m   303    583    669     0
  B5ED    360       54        5.2 m    16.8 m    67.2 m   119    214    291     3

Reading: the margin-0 box clipped legitimate near-edge fixes on ALL four tags
(41-62% of dropped fixes within 5 m of the edge, 81-92% within 25 m). The
inside-box "drops" (6/0/58/54) are consensus clusters straddling the edge —
the 10 m consensus radius picks up members outside the box, so the median can
sit just outside even when most evidence is inside; a margin fixes that too.
No dropped fix is beyond ~50 m (3 B5ED at 50-67 m): km-scale jump artifacts
are already rejected at the solution level by the box, as designed — the
consensus-level cost of margin 0 is pure near-edge clipping. A fish
legitimately sits a few metres outside the outermost receiver.

## Parameter change (no-silent-tuning rule)
- position_3d_box_margin_m: 0.0 -> 10.0 in BOTH run files:
  - config/run_data_earlyfish_2025.toml (canonical study file; key added
    2026-11-14)
  - config/run_data_earlyfish_2026-10-28.toml — the run file the on-disk Arm A DB
    actually references (its
    jsats3d_2025_earlyfish.pipeline_status.json points at it), and the one
    fish_dashboard_v3 auto-resolves via run_file_from_status(). Without the
    keys there, the dashboard would run with box OFF and its self-check
    would abort. Both files now carry the identical [figures] block with the
    evidence citation.
- Choice: 10 m = the EXISTING consensus radius (legacy_receiver_set_consensus
  10 m solution cluster radius) — reused, not fitted. At 10 m the window is
  X [427.93, 518.56], Y [69.52, 145.28], Z [-16, 0]. It recovers 60-80% of
  the clipped fixes per tag (26E4 103/127, 9F8C 16/34, 1D5A 583/730, B5ED
  214/360) while still rejecting everything >=10 m outside the edge: the
  26E4 jump cluster (~50-150 m NW of the array) and the X~570 m excursions
  remain rejected. A 25 m margin would recover 81-92% instead; noted as the
  next step if the owner wants more permissiveness (no additional gross
  errors would be admitted at 25 m either, per the distance table).

Margin=10 4-tag pass (final, overwrote output/pm_figures_2026-11-15/example_fish/):

  tag   solns  above  floor  box   consensus  smooth  kalman  presence(outside/notcon/nosol)  max_v  jumps>3
  26E4  19897   5457    685   238      1377       4     1373     1604 (81/517/1006)          2.19     0
  9F8C   1654    759    167   147        57      12       45     1612 (40/85/1487)          1.45     0
  1D5A  51859  26226   1884   790      3375      42     3333    13601 (1011/3169/9421)     11.49     3
  B5ED  37263  11434   4118  1461      2386      89     2297     7654 (514/1746/5394)       4.62     4

Kalman counts vs the pre-box contract (fish_qc 2026-10-25): 26E4 1387->1373
(99.0%), 9F8C 65->45 (69%), 1D5A 3413->3333 (97.7%), B5ED 2327->2297 (98.7%).
The residual is the gross rejection the filter exists to do (9F8C is the
small denominator: 20 gross fixes out of 65).
Observation (pre-existing, NOT caused by the box — present in the old chain
too): 1D5A max speed 11.49 m/s with 3 jumps >3 m/s (06-19 03:55-03:56 local,
1-0 m moves in 0 s — sub-second same-position hops, likely a
metronome/detection artifact at that time) and B5ED 4 jumps of 11-14 m in 3 s
(06-26, plausible darting at Z -0.65 m). Flagged for the PM; no action taken
this session.

## Task 2 — fish_dashboard_v3.py port
- Copied byte-exact from the frozen branch: git checkout
  ENM_jsats3d_edits_clocksync -- scripts/fish_dashboard_v3.py; verified with
  git hash-object == git rev-parse <branch>:<path> (aed6f9494a4ca3203781f92fe6899e96db18039e).
  (An earlier git show | Out-File attempt mangled line endings — 835 vs
  expected lines; git checkout fixed it. The frozen branch itself was not
  modified by the checkout.)
- The file as committed on clocksync contains mojibake ("ΓÇö" where em-dashes
  should be — cp437 double-encoding from when it was created). Left as-is;
  new edits use ASCII.
- Changes made on THIS branch (5):
  1. Docstring: chain now includes the 3D window screen step; self-check
     contract updated with the 2026-11-15 numbers and the provenance (Arm A
     DB, margin 10 m); usage line gained [--run-file TOML].
  2. Imports: +tomllib; ent_analysis import extended with
     receiver_3d_box, three_d_box_mask.
  3. EXPECTED_KALMAN: {"26E4": 1373, "9F8C": 45, "1D5A": 3333, "B5ED": 2297}
     with the old contract (1387/65/3413/2327) kept in the comment.
  4. load_track(db, tag, floor_m, box=None, box_margin_m=0.0): after the
     Z gate, applies three_d_box_mask (np.asarray — it returns an ndarray)
     when box is given; returns a 4-tuple (gk, n_raw, n_gated,
     n_outside_box). box=None => legacy chain bit-identical.
  5. main(): --run-file arg (default: auto via run_file_from_status); reads
     [figures] position_3d_box/position_3d_box_margin_m and [legacy]
     receivers_3d exactly as fish_figures_2025.py does; builds the box from
     tblReceiver (same fallback note when a configured receiver is missing);
     prints the window; passes box+margin to load_track; per-tag line now
     shows outside_box count. README text (Track chain section) updated.
- Run (auto-resolution would have worked too, but --run-file passed
  explicitly for determinism):
  python scripts/fish_dashboard_v3.py --db output/10292026_Test/jsats3d_2025_earlyfish.db \
      --out output/pm_figures_2026-11-15/fish_dashboard_v3 --run-file config/run_data_earlyfish_2026-10-28.toml
  -> 3D window printed (margin 10 m, 9 receivers); all four tags
  "OK (matches verified fish_qc)"; wrote dashboard.html (6.2 MB),
  fish_tracks.csv (7048 rows), fish_roster.csv, README.md. Fates: 26E4
  FOREBAY-LOST, 9F8C FOREBAY-LOST, 1D5A REACHED DAM (PIT 6/19 10:06 @ CFD),
  B5ED FOREBAY-LOST. Old build at output/pm_figures_2026-10-22/fish_dashboard_v3/
  preserved untouched (old chain).
- Known wart (not changed): DEFAULT_DB = output/jsats3d_2025_earlyfish.db no
  longer exists on disk, so a bare `python scripts/fish_dashboard_v3.py`
  fails until --db is passed. Left byte-faithful to the ported default;
  owner may repoint it.

## Gotchas hit (new)
- PowerShell `cmd > file.log` redirects in UTF-16LE — the read tool then
  refuses the file as binary. Convert (Get-Content -Raw | Out-File -Encoding
  ascii) or run the process without redirection.
- `git show branch:file | Out-File` is NOT byte-exact (line-ending
  mangling); use `git checkout branch -- path` (updates working tree + index,
  does not touch the source branch) and verify with git hash-object.
- fish_dashboard_v3.py continuation-line indentation is 36 spaces (aligned
  to the opening paren); an edit with 37 spaces silently fails to match.
- The Arm A folder's pipeline_status.json references
  config/run_data_earlyfish_2026-10-28.toml, NOT run_data_earlyfish_2025.toml.
  Any run-file-dependent tool that auto-resolves from the Arm A DB (the
  dashboard) needs its [figures] keys in the 10-28 file.

## Blockers
None.

## Next steps
1. Owner commits on ENM_jsats3d_clock_updates (nothing committed by this
   session, per project rule): LONG_TERM_CONTEXT.md, both run files,
   scripts/fish_dashboard_v3.py (new here), .ai_journal (this file),
   output/11152026_4tag_margin/ + 11152026_4tag_pass.txt (evidence),
   output/pm_figures_2026-11-15/ (4-tag figures + dashboard).
   (Uncommitted from the 2026-11-14 session still rides along:
   ent_analysis_2025.py, fish_figures_2025.py, position_box_compare_2025.py,
   tests/test_position_box.py, 2026-11-14 journal.)
2. PM review: dashboard.html (pm_figures_2026-11-15/fish_dashboard_v3/),
   4-tag fish_qc.csv, and the 26E4 box compare plot (11142026_box_compare).
   Ask the PM whether 25 m margin is wanted instead (more near-edge recovery,
   no gross-error cost per the distance table).
3. Investigate the 1D5A 06-19 03:55-03:56 sub-second same-position hops and
   B5ED 06-26 dart jumps (pre-existing, see observation above) if the PM
   wants clean speed stats.
4. 2027 items unchanged: TDOA jump detection from the frozen clocksync branch
   as the pre-deployment check; present_unpositioned visualization
   (Kevin: "save that for later").
