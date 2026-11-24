# 2026-11-23 - step-aware clock filtering and low-support position flags

## Request
- Implement the step-aware clock filter tested locally on 26E4 (06-28 00:00-02:00) as an explicit 2025 option:
  - endpoint retention
  - jump detection before filtering
  - one DBSCAN per stable segment
  - bounded 2/1/0-point fallback
  - midpoint holds
- Flag Deng positions that use a receiver whose clock segment has fewer than 3 beacon observations; propagate the
  flags.
- New DB in `output\clockskip`, plus the two reference figures.
- Builds on Drew's commit d006c47 (`segmented_clock_dbscan`, per-segment DBSCAN, endpoint, midpoint holds).

## Code
- `jsats3d/jsats3d.py` (only runs when `segmented_clock_dbscan` is set; the five-pass path is unchanged):
  - `_drop_isolated_clock_spikes`: a level of 1-2 observations that steps away and back is removed. Its neighbours
    must agree within `clock_step_s`, with no outage on either side. A short level between different levels is kept
    (multi-step resync).
  - `_segmented_clock_dbscan`:
    - Boundaries are `step` (held to the midpoint), `outage` (no step, not bridged) or `step_outage` (step with a gap
      over `clock_jump_max_gap_s`, not held).
    - Support and method per level: supported (>= 3), linear (2), constant (1), none (0). 1-2 are allowed only for
      jump-related levels; non-jump short levels are dropped.
    - Optional `summary` list: provenance per level and per removed spike.
  - Rows carry `clock_segment_support` and `clock_segment_method`.
  - `_interpolate_clock_segments(..., return_support=True)` returns the support per time.
  - The fish correction (multipath_data_object) adds a `clock_support` column per detection. It flows to
    tblDetectionFilterPrimary/Secondary and Deng `tag_data`.
  - `clock_fix` passes `clock_step_s` / `clock_jump_max_gap_s` and keeps `clock_segments` on the object. The master
    is one 'master' segment.
- `scripts/legacy_pipeline.py`: `clock_fix` / `clock_fix_check` accept the options and write `tblClockSegments`,
  replacing rows per receiver. `tblClockSegments` is added to DERIVED_TABLES (dropped and rebuilt with the clock).
- `scripts/_2025_pipeline.py`:
  - `clock_options`: `[ats_2025_pipeline] segmented_clock_dbscan`, `clock_step_s` (0.05), `clock_jump_max_gap_s` (600).
  - `annotate_clock_segments`, run after steps 4 and 7, records evidence per step:
    - `internal_event`: OffsetChanged/CounterRestart in tblDetectionRaw.
    - `cross_receiver_steps`: other receivers stepping in the same interval.
  - `flag_clock_support`: Deng CSVs, both roots, get `clock_low_support`, `clock_low_support_receivers` and
    `clock_min_support` from the receivers actually in r0-r3. Rows are kept; export, premedian and position DBSCAN
    (`p.*`) carry the flags.
- `scripts/ent_analysis_2025.py` `legacy_receiver_set_consensus`: when present, aggregates the flags over the
  contributing (nearby) solutions only (`clock_low_support`, receivers, `clock_low_support_solutions`, min support).
- `scripts/fish_figures_2025.py`: reads the flag columns if present; `fish_positions.csv` appends them;
  `fish_qc.csv` gains `n_consensus_clock_low_support`.
- `README.md`: option section rewritten.
- Tests:
  - `tests/test_clock_segments.py`: 26E4 midpoint holds (01:01:57 / 01:04:04), linear 2-point fallback, isolated
    spike, unsupported short level, step across a long gap, and flag propagation through the CSV and consensus.
  - `tests/test_smoke.py`: Drew's one-knot test changed to a multi-step resync. The old 0-1-0 case is now isolated
    multipath and is tested separately.
  - 19 pass (jsat_3d pytest); unittest is OK in the cpu env.
- `tests/clock_segment_comparison.py` (test tool): the plan and event figures, plus `gap_recovery.csv`,
  `jump_events.csv`, `normal_periods.csv` and `receiver_timing.csv`.

## Run (`output\clockskip`)
- Configs `config/run_data_clockskip_{base,baseline,segmented}.toml`: copies of `run_data_earlyfish_2026-10-28.toml`
  (4 fish, 06-15..08-06, 9 ZOI 3D receivers, no premedian/cap) with raw paths under C:\jsats_data.
  - base: build only, 20 min, 13.6 GB.
  - The arms are copies of the base DB run with `_2025_pipeline.py --no-drag-plots`, then fish figures.
- Run times:
  - baseline pipeline about 40 min.
  - segmented about 1 h 45 min: clock_surface 28.8 min vs 6.3, deep_clocks 42.5 vs 13.7. The spike loop regroups
    after every removal; it could be optimised.
- Segments: 41,297 levels (deep run). 1,896 held steps, 762 steps beyond 600 s, 856 isolated spikes, 517 fallback
  levels, 7,981 unsupported levels. 42,479 outage boundaries.

## Results (`output\clockskip\comparison`)
- Known 26E4 gap (00:54:59-01:05:47):
  - Root-B transmissions 5 -> 156 (reference 4 -> 166); 4 -> 124 after position DBSCAN (reference 148).
  - In water 40% -> 53%; 32 flagged (the single-beacon holds 01:02-01:05, several implausible, Z -13..+4).
  - The event is seen on 15 receivers (master-clock resync); 6 had internal events.
- Other events:
  - 1D5A 06-18 23:56 (ZOI10 -1.0 s): 82 -> 192, in water 13% -> 44%.
  - 9F8C 06-26 15:47: 0 -> 53.
  - B5ED 06-26 14:47 (ZOI05 +1.0 s for about 10 min, then back): 40 -> 71, but in water 42% -> 23% and speed jumps
    4 -> 29. That held level gives implausible positions above the surface.
- Degradation outside jump windows (main issue):
  - First-arrival detections: 1D5A 159,707 -> 53,258, B5ED 79,026 -> 47,517, 9F8C 13,247 -> 6,343,
    26E4 30,301 -> 20,282.
  - Positioned transmissions in normal periods: 1D5A 4,668 -> 2,405, B5ED 3,869 -> 2,903, 9F8C 173 -> 102.
    fish_positions: baseline 7,412 positioned.
  - Cause: every beacon gap > 3 periods (192 s) with no step is an `outage` boundary and is not bridged. Fish
    detections in it are excluded; 4-95% of time is unsupported per receiver (ZOI06 95%, CFD06 75%).
  - Where both arms have a correction, the corrections are identical (median |diff| 0.0 ms). Position differences on
    matched pings come from the changed receiver sets.
- Open decision for the project lead: bridge step-free outages explicitly (linear, with provenance and a bound), or
  keep the strict exclusion.
- fish_positions.csv positioned (consensus): baseline 7,412 -> segmented 4,253 (26E4 1,398 -> 1,292; 1D5A 3,465 -> 1,366; B5ED 2,473 -> 1,523; 9F8C 76 -> 72). Segmented flagged positioned rows: 149 (clock_low_support = 1).
