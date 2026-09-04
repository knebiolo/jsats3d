# Acoustic Positioning Workflow

This document outlines the step-by-step workflow for setting up, importing, filtering multipath, synchronizing receiver clocks, positioning submerged receivers at depth, and calculating 3D positions for tagged fish using `jsats3d`.

---

## Workflow Steps

### Phase 1: Initial Receiver & Surface Alignment
1. **Project Setup**:
   - Initialize project database schema via `jsats3d.create_project_db()`.
   - Populate `tblStudyParameters`, `tblTag`, `tblReceiver`, and `tblWSEL`.
   - Import raw receiver detection files using `jsats3d.acoustic_data_import()`.

2. **Temperature Assessment**:
   - Create and store continuous water temperature time-series in `tblInterpolatedTemp`.
   - Instantiate temperature interpolator via `jsats3d.temp_interpolator()`.

3. **Metronome (Surface Receivers)**:
   - Perform beacon epoch transmission enumeration limited to surface receivers whose positions are known via GPS.

4. **Clock Fix Serial (Surface Receivers)**:
   - Perform linear drift clock correction for surface receivers relative to the master clock receiver.

5. **Multipath Filter (Submerged Receiver Beacons)**:
   - Filter multipath detections on beacon tags transmitted by submerged receivers at depth.

6. **Coordinate With Deng (Submerged Receivers)**:
   - Calculate exact 3D positions of submerged receiver beacons using Daniel Deng's 3D TDOA positioning solver.
   - Calculate median X, Y, Z position for each submerged receiver across all valid solutions.
   - Update master `tblReceiver` table with finalized receiver coordinates (`X_t`, `Y_t`, `Z_t`).

---

## Phase 2: Full Study Tag Positioning
7. **Metronome (Submerged Receivers)**:
   - Perform beacon epoch transmission enumeration across all resolved submerged receivers.

8. **Clock Fix Serial (All Receivers)**:
   - Execute final clock drift corrections across all receivers in the study array.

9. **Multipath Filter (All Study Tags)**:
   - Train and run multipath classifiers (SVM / Naive Bayes / Decision Tree / KNN / DBSCAN) across all study tags.

10. **Final Positioning & Export**:
    - Run 3D Deng positioning solver for all study tags.
    - Export trajectory points, calculate kernel utilization distributions (`jsats3d.kernels`), and prepare final datasets.
