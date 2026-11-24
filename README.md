# jsats3d

jsats3d was developed by Kleinschmidt to assist researchers with cleaning and managing small scale acoustic telemetry studies (<1000 tagged individuals, <20 receivers) The resulting dataset represents a standardized, repeatable and defensible product. This notebook carries out every task one may have when analyzing acoustic telemetry data, including: creating a standardized technology agnostic project database, importing raw data directly from receiver downloads, identifying and removing multipath, synchronizing clocks, positioning fish in 3D space, and visualizing resultant point clouds with kernel density functions. In future iterations of the software, we will replace the functional sample scripts with a Jupyter Notebook to guide users through training, classification, and data preparation.

Just open up the Project Notebook and get started

For the 2025 step-aware clock filter, set `segmented_clock_dbscan = true` in
`[ats_2025_pipeline]` (default false keeps the existing five-pass 2019/2025
behavior). The beacon residual curve is split at changes of at least
`clock_step_s` (0.05 s, a Cowlitz 2025 starting point, not a universal
threshold), after removing isolated multipath (a one- or two-observation
excursion that returns to the same level). A step-free beacon gap does not
split the curve; the correction is linearly interpolated across it, even when
the gap is long and unobserved clock changes may have occurred.
DBSCAN (`min_samples = 3`) runs once per segment with the actual final timestamp
included. Within multi-point segments fish corrections use linear interpolation.
At a step whose beacon gap is at most `clock_jump_max_gap_s` (600 s), each side
holds its endpoint residual to the midpoint between the last pre-step and first
post-step observation; these midpoint jump times are estimates, not measured
events. Segment support: three or more retained beacon observations is normal;
two (linear) or one (constant) is accepted only next to such a step; anything
else is unsupported, and fish detections in unsupported time and steps across
longer gaps are excluded rather than bridged. `tblClockSegments` records
every segment's support, boundaries, fallback method, whether its boundaries are
estimated, and step evidence (receiver internal events, steps on other receivers).
Deng rows whose receiver combination uses a segment with fewer than three
observations get `clock_low_support`, `clock_low_support_receivers` and
`clock_min_support`; the flags carry through export, position DBSCAN, the
per-transmission consensus and `fish_positions.csv`. Flagged positions are kept.
Use a separate database and regenerate clock-fixed and downstream results when
changing the option. `tests/clock_segment_comparison.py` compares a baseline and a
segmented copy of one build.
