# 3D Window Screen + Present-But-Unpositioned Export — Session Journal
# Session: 2026-11-14
# Branch: ENM_jsats3d_clock_updates (new; based on ENM_jsat3d_edits @ b332c00)

## PM request (message thread 2026-11-13, Kevin + Ethan)
1. Drop the clock-jump corrections; instead apply a 3D filter on clusters of
   solutions BEFORE taking the median for the final selected position.
   Reuse existing geospatial-filter thresholds as a start; worst case the window
   is the min/max of receiver positions in XYZ.
2. Accept that time jumps occur and data is lost during them: the exported data
   must indicate a fish is within the 3D area but not positioned (so in-array
   fish-hours don't read as the fish leaving and re-entering). Visualization
   deferred.
3. After implementing: compare a fish where a time jump occurs, with vs without
   the filter, in the same plot.
4. Context for the January 2027 meeting: firmware update (Mark/Sheat has all
   firmware), deploy 1-2 weeks early, run TDOA analysis to identify time jumps
   pre-deployment; positions incl. depth receivers validated by end of
   preliminary study stage.

## Decisions
- Work happens on NEW branch ENM_jsats3d_clock_updates (owner created it at the
  ENM_jsat3d_edits base before this session). ENM_jsats3d_edits_clocksync is
  untouched: frozen, never merged (owner committed its C5 files there as bea2bf3
  overnight). No commits made on any branch by this session — owner commits.
- The 3D window is X/Y = receivers_3d min/max (from the run file [legacy],
  falling back to all tblReceiver rows with a printed note when a configured
  receiver is missing), Z = the EXISTING [depth floor, 0] plausibility window,
  NOT the receiver Z extent: fish swim above the shallowest hydrophone (26E4
  measured Z -0.2 m; shallowest hydrophone -2.26 m). Reusing the existing Z
  thresholds is exactly the "use the existing thresholds as a start" ask.
- The screen is applied in the product chain (fish_figures_2025.py), the same
  place the existing Z plausibility gate lives — NOT inside
  legacy_receiver_set_consensus, so the GPS-verified drag scoring chain
  (drag_test_figures_2025.py baseline contract) and ent_analysis item12 stay
  bit-identical.
- Rejected solutions are FLAGGED (counted in fish_qc.csv n_outside_box, printed
  per tag), never silently deleted; they are not written to the client CSV
  (processing artifacts, same policy as the Z gate).
- Presence definition: an emission (transNo) with >=1 first-arrival detection on
  the receivers_3d set and no final consensus position. Reasons:
  outside_3d_box (every solution fell outside the window), not_consensus
  (solutions existed, none survived consensus), no_solution (no root-B solution).
  receiver_sets = distinct 3D receivers hearing the emission (tightening is
  always possible downstream; widening after export is not, so >=1 is the safe
  default).
- Presence rows go in the SAME <TAG>_positions.csv (product =
  'present_unpositioned', XYZ empty/NaN — never fabricated — plus a new
  reason column, empty on position rows) because the point is that the primary
  timeline must show in-array hours, not a side file that must be joined.
- Defaults: position_3d_box = false (opt-in, legacy bit-identical until
  validated — then flipped true in run_data_earlyfish_2025.toml with this
  journal's numbers as the rationale); present_unpositioned = true (it is the
  requirement, not an experiment). CLI overrides: --box on/off/auto,
  --box-margin-m, --presence on/off/auto.

## Files touched
- scripts/ent_analysis_2025.py: added receiver_3d_box(), three_d_box_mask()
  (NaN coordinates rejected, inclusive bounds, margin on all axes),
  present_unpositioned_rows(). legacy_receiver_set_consensus unchanged.
- scripts/fish_figures_2025.py: docstring (chain + products + config keys);
  [figures]/[legacy] run-file reading via tomllib; --box/--box-margin-m/
  --presence args; per-tag box screen before consensus; box-rejected transNo
  set (groupby-any, O(n)); presence from tblDetectionFilterSecondary
  (fallback tblDetectionRaw with a printed note); QC columns n_outside_box +
  n_present_unpositioned + per-reason counts; reason column + presence rows in
  <TAG>_positions.csv.
- scripts/position_box_compare_2025.py: NEW persistent comparison tool (PM's
  with-vs-without plot): shared Z gate, then consensus without box vs with box;
  plan full-window + plan zoom (median rejected emission +-2 h) + X(t) + Z(t);
  metrics CSV (fixes dropped/new, matched drift, gross = out-of-window fix
  counts); row-level g_off/g_on/rejected CSVs.
- config/run_data_earlyfish_2025.toml: [figures] position_3d_box (true, with
  validation citation), position_3d_box_margin_m = 0.0, present_unpositioned
  = true.
- tests/test_position_box.py: NEW, 15 tests (box min/max + NaN + empty; mask
  inside/outside/boundary/margin/NaN; majority-gross consensus breaks without
  the screen and is restored with it; single-outlier sanity; legacy default
  unchanged; presence reasons/omissions/empty/priority).
- LONG_TERM_CONTEXT.md: 2026-11-14 standing decision row.

## Parameter changes (journal per the no-silent-tuning rule)
- position_3d_box: false -> true in run_data_earlyfish_2025.toml only. Rationale
  = validation below. No other run file touched.
- position_3d_box_margin_m = 0.0: no margin to start; the receiver min/max is
  the physical boundary. If the box clips legitimate near-edge fixes, validation
  will show how many and how far, and a margin gets added with evidence.
- present_unpositioned = true (new behavior, new CSV rows; position rows and
  all existing columns unchanged except the added reason column).

## Validation (Arm A unshifted DB output/10292026_Test/jsats3d_2025_earlyfish.db)
Fish: 26E4, the June 27-28 ZOI01/02/03/06 jump cluster night (all 19,897 root-B
solutions in 06-27..29 fall in the 27 23:15 -> 28 01:30 local transit).
Artifacts: output/11142026_box_compare/ (26E4_box_compare.png,
26E4_box_compare_summary.csv, g_off/g_on/rejected CSVs, figures/ from a
full fish_figures run with --box on).
- 19,897 root-B solutions -> Z gate drops 5,457 above-surface + 685
  below-floor -> 13,755 -> 3D box rejects 1,080 (7.8%).
- Consensus WITHOUT box: 1,398 fixes, 157 of them (11.2%) OUTSIDE the array 3D
  window (physically impossible: the fish is tracked by receivers spanning a
  ~70x55x16 m volume).
- Consensus WITH box: 1,271 fixes, 0 outside the window.
- 1,271 matched transmissions: drift median 0.0 m / p99 8.67 m / max 14.2 m —
  the box does not move valid fixes.
- 127 fixes dropped (every candidate for those emissions was outside the
  window) -> they surface as present_unpositioned rows (321 outside_box total
  including no-solution emissions).
- Full fish_figures run (--box on): kalman=1,268, max speed 1.40 m/s,
  0 jumps > 3 m/s (the 2026-10-24 first-pass phantom speeds were 53,351 m/s
  class); CSV = 1,271 consensus + 1,268 kalman + 1,710 present_unpositioned
  (1,006 no_solution / 383 not_consensus / 321 outside_3d_box).
- The plot (26E4_box_compare.png) shows the without-box consensus snapped to a
  cluster ~50-150 m NW of the array (around (450,145)) and X excursions to
  ~570 m during the jump interval; the with-box track stays inside the window
  for the whole transit.
- Tests: conda env jsat_3d python -m unittest discover -s tests -> Ran 21
  tests, OK (6 pre-existing + 15 new).

## Gotchas hit
- three_d_box_mask returns an ndarray, not a Series — fish_figures had to
  np.asarray before wrapping in pd.Series (caught in code review, not runtime).
- The "gross" metric must be "outside the receiver window", NOT
  |coord| > 300: the local coordinate origin is far from the array
  (receivers at X 437-508, Y 36-139), so absolute-coordinate thresholds
  mislabel every legitimate fix. (An earlier probe using that threshold
  wrongly suggested "all 19,897 solutions gross".)
- pandas: df.product resolves to the DataFrame.product METHOD, not the column —
  use df["product"].
- This env's pandas is 1.x: pd.to_datetime has no tz= kwarg (use utc=True), and
  datetime Series need .dt.strftime.
- "26E4" is parsed as the float 260000.0 (26e4) by naive CSV readers — known
  project trap; consumers must use dtype={"tag": str}.

## Blockers
None.

## Next steps
1. Owner commits on ENM_jsats3d_clock_updates (nothing committed by this
   session, per project rule).
2. Run the full 4-tag fish_figures pass with the new run file (box on +
   presence) and review fish_qc.csv per tag before it goes to the PM; check
   whether any tag shows clipped near-edge fixes (would justify a margin).
3. PM review of 26E4_box_compare.png (the with/without plot Kevin asked for).
4. 2027: reuse the FROZEN clocksync branch's TDOA jump DETECTION (stage 1/2)
   as the pre-deployment receiver check; firmware update + early deployment per
   the January meeting plan (LONG_TERM_CONTEXT 2026-11-14 row).
5. Visualization of present_unpositioned hours (Kevin: "save that for later").
6. Decide whether fish_dashboard_v3 (clocksync branch) should be cherry-picked
   to this branch when the dashboard needs the new products — deferred, since
   the owner said to only work in this branch and the dashboard is not part of
   the client _positions.csv chain.
