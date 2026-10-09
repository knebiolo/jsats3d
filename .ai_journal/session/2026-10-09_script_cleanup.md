# Session: 2026-10-09 Script Cleanup

**Author:** AI assistant
**Timestamp:** 2026-10-09
**Tags:** cleanup, scripts, tests
**Status:** Complete

## Files Touched
- `scripts/cfd_gps_diagnostics.py` — deleted (diagnostic only, not on the pipeline path).
- `tests/test_2025_contracts.py` — removed the `cfd_gps_diagnostics` import and the test `test_piecewise_gps_interpolation_supports_linear_cubic_and_no_extrapolation`.

## Decisions
- Kept (pipeline-critical): `tagdrag_2025_pipeline.py`, `legacy_pipeline.py`, `run_data.py`, `parse_ats_raw_to_legacy.py`, `adapt_2025_to_legacy.py`, `beacon_pairwise_dbscan.py`, `ent_analysis_2025.py`.
- Kept (Kevin's reference drivers, not ours): `clock_fix_serial.py`, `metronome.py`, `mulitpath.py`, `coordinate_with_Deng.py`, `tag_drag_RMSE.py`.
- Kept by owner choice: `extract_dbscan_features.py` (imported by `tests/test_2025_contracts.py`; 09-23 decision).
- Output logs and diagnostic CSVs in `output/` were not deleted.

## Verification
- Test suite run with `envs\jsat_3d\python.exe -m unittest discover -s tests -q`: exit 0.
