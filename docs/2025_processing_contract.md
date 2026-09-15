# 2025 Processing Contract

Status: design contract. No production filtering or positioning parameters are approved.

## Purpose

Define stable inputs, outputs, provenance, and accounting for the 2025 ATS pipeline while preserving legacy-compatible table names where useful.

## Stage Contracts

| Stage | Required inputs | Required outputs | Acceptance evidence |
|---|---|---|---|
| Schema audit | Raw deliverables | Schema report | Columns, timestamp precision, signal-field inventory |
| Ingestion | Raw or documented-clean detections, receiver metadata, environment data | `tblTag`, `tblReceiver`, `tblDetectionRaw`, `tblInterpolatedTemp`, `tblWSEL`, `tblStudyParameters` | Row reconciliation, NULL report, coordinate provenance |
| Beacon coverage | Beacon detections, beacon registry | Beacon coverage CSV | Tag/receiver counts, date span, configured-vs-observed reconciliation |
| Epoch construction | Beacon detections, verified transmission periods | Epoch table | Period source, assignment counts, unassigned counts |
| Multipath filtering | Epoch detections, amplitude, approved feature parameters | Filter result table | Input/output counts, rejection reasons, parameter record |
| Clock verification | Filtered beacon epochs, receiver geometry, sound speed | Clock residual table and plots | Offset, drift, residual distribution, time span, warning flags |
| Positioning | Clock-corrected detections, receiver geometry, temperature/WSEL | Position table | Four-or-more receivers per epoch, solution status, hull status |
| Validation | Positions, surveyed holds or drag tracks | RMSE/precision/yield reports | Truth source, shared validity mask, inside/outside hull metrics |

## Detection Provenance

Every derived detection record should retain:

- `source_file`
- `source_row` or stable source identifier
- `source_processing_status`
- `receiver_type`
- `coordinate_source`
- `filter_stage`
- `rejection_reason`
- `run_id`

Unavailable physical measurements remain explicit `NULL`. No synthetic SNR, NBW, frequency offset, pressure, or receiver temperature values are permitted.

## Multipath Feature Design

Approved candidate features, calculated per tag, receiver, and transmission epoch:

1. `lag_seconds`: detection time minus earliest detection in epoch.
2. `relative_amplitude`: amplitude minus epoch maximum, or an equivalent explicitly documented relative transform.
3. `epoch_rank`: ordinal detection order within epoch.
4. `inter_detection_seconds`: interval to neighboring detections of the same tag and receiver.

Features must preserve source rows and expose missing or undefined values. Absolute amplitude must not be compared across receiver models without gain and threshold metadata.

DBSCAN parameters remain unapproved. The prior exploratory `eps` and `min_samples` values must not be copied into 2025 processing.

## Rejection Accounting

Each run must report:

- Raw rows read.
- Rows parsed.
- Rows dropped for invalid timestamps.
- Rows dropped for missing identifiers.
- Rows assigned to epochs.
- Rows rejected by each filter stage.
- Rows retained.
- Rows eligible for positioning.
- Positions produced.
- Positions retained outside the convex hull.
- Rows rejected for each explicit reason.

Counts must reconcile. A row may have one primary rejection reason per stage, and stage totals must not silently overlap.

## Static-Hold Report

Minimum fields:

- `test_id`
- `start`
- `end`
- `tag_id`
- `receiver_count`
- `detection_count`
- `position_count`
- `rmse_x_m`
- `rmse_y_m`
- `rmse_z_m`
- `precision_x_m`
- `precision_y_m`
- `precision_z_m`
- `inside_hull_count`
- `outside_hull_count`
- `truth_source`
- `coordinate_source`
- `run_id`

No RMSE is valid unless the comparison uses a shared timestamp and coordinate validity mask.

## Beacon Coverage Report

Minimum fields:

- `beacon_tag_id`
- `configured_receiver`
- `configured_period_seconds`
- `detection_rows`
- `detection_receivers`
- `first_seen`
- `last_seen`
- `period_source`
- `assignment_status`
- `coverage_status`

Broad receiver coverage does not by itself prove array-wide beacon identity or synchronization suitability.

## Output Compatibility

Legacy-compatible names may be retained for interoperability:

- `tblDetectionRaw`
- `tblMetronomeUnfiltered`
- `tblMetronomeFiltered`
- `tblMetronomeSecondFiltered`
- `tblDetectionFilterPrimary`
- `tblDetectionFilterSecondary`
- `tblDetectionClockFixed`
- `tblPositions_Deng`

Compatibility means table shape and traceable semantics. It does not mean legacy KNN behavior, legacy synchronization assumptions, or identical scientific results.

## Hard Stops

Do not produce accepted 2025 positions when any of these remain unresolved:

- Sound-speed temperature coverage.
- Coordinate authority and vertical datum.
- Beacon identity and period.
- Synchronization architecture.
- Raw-versus-cleaned detection provenance.
- Required geometry for the evaluated epoch.

## Synchronization Readiness

`jsats3d.sync_readiness.assess_sync_readiness()` is a preflight gate. It checks:

- Required detection, receiver, temperature, and beacon-epoch columns.
- Complete receiver geometry count.
- Beacon epoch receiver coverage.
- Temperature coverage across detection times.

It does not estimate clock offset, clock drift, sound speed, beacon period, or residual thresholds. It must report `ready=False` when required inputs are absent.