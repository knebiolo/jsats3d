# 2025 Owner Input Checklist

Status reflects local data audits through 2026-09-15.

## Confirmed Locally

- Detection files use `dateTime`, `tagCode`, `amp`, `receiverName`, and `event`.
- Sampled timestamps contain six fractional digits.
- FFD3 observed interval is approximately 3.34 seconds across receivers.
- FFD3 validation windows exist in the controlled-test workbook.
- Forebay static holds DB-1 through DB-5 provide the strongest current FFD3 coverage.
- Current FFD3 staging has 23 receivers with complete X/Y/Z values.
- Six active beacon IDs are broadly detected but lack local receiver assignments: `1F14`, `1F38`, `1F5A`, `1F71`, `1F94`, and `1FCD`.
- Configured beacon IDs `2008` and `2010` have no detections in the beacon deliverable.
- Temperature and WSEL data begin June 17, after the June 5-16 FFD3 detections.

## Owner Confirmation Required

### Study and Datums

- [ ] Confirm job number and deployed study scope.
- [ ] Confirm raw-data root and authoritative deliverable version.
- [ ] Confirm benchmark elevation and vertical datum.
- [ ] Confirm coordinate datum and projection for supplied receiver coordinates.
- [ ] Confirm whether hydrophone depths are referenced to benchmark, water surface, or another datum.

### Receiver Inventory

- [ ] Confirm active receiver list for FFD3 validation and full study.
- [ ] Confirm receiver model and clock behavior for each active unit.
- [ ] Supply missing CHN and DNS horizontal coordinates, or approve their exclusion.
- [ ] Confirm whether CFD05 and CFD09 should be included in full-study processing despite lacking FFD3 detections.
- [ ] Supply gain, sensitivity, and threshold settings if amplitude comparisons across units are required.

### Beacons and Synchronization

- [ ] Identify array-wide beacon tag IDs.
- [ ] Confirm transmission periods for `1F14`, `1F38`, `1F5A`, `1F71`, `1F94`, and `1FCD`.
- [ ] Explain duplicate configuration rows for the six unassigned beacon IDs.
- [ ] Explain absent detections for `2008` and `2010`.
- [ ] Confirm whether local receiver beacons and array-wide beacons were active simultaneously.
- [ ] Provide the intended synchronization architecture and parameters.
- [ ] Confirm GPS synchronization status for SR3017 units.
- [ ] Confirm internal-clock behavior and deployment timing for autonomous units.

### Environmental Data

- [ ] Supply authoritative temperature data covering June 5-16, or approve a documented alternate source.
- [ ] Supply WSEL data covering June 5-16 if required for the controlled-test positions.
- [ ] Confirm whether the temperature string or another source controls sound speed.
- [ ] Confirm sound-speed formulation and required depth treatment.

### Detection Provenance

- [ ] Supply raw, unfiltered detections in addition to cleaned deliverables.
- [ ] Document any upstream multipath removal.
- [ ] Confirm whether frequency and background-noise fields were dropped during concatenation.
- [ ] Confirm whether amplitude units are dB and explain zero-amplitude records.

## Processing Gate

Do not run accepted clock correction or positioning until the synchronization, datum, and June environmental coverage items are resolved. Provisional diagnostics may continue, but every result must be labeled as provisional and must not be used as a Gate 2 or Gate 4 acceptance result.

## Local Artifacts

- `output/detection_schema_audit_all_samples.txt`
- `output/beacon_coverage_audit.csv`
- `output/arraywide_beacon_intervals.csv`
- `output/beacon_amplitude_audit.csv`
- `output/receiver_3d_readiness.csv`
- `output/ffd3_validation_windows.csv`
- `output/2025_preflight_audit.txt`