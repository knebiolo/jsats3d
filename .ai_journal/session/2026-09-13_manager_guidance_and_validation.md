# Session: 2026-09-13 — Manager Guidance & Validation Strategy

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-13 UTC
- Tags: #manager-guidance #parity #dbscan #validation #branch-safety #visualization

## Active Context
- Files updated: `System_Prompt.txt`, `LONG_TERM_CONTEXT.md`
- Manager guidance source: user-provided project discussion
- Current focus: Translate production direction into standing project rules without changing legacy code

## Summary
Manager guidance clarified project direction. This work is production implementation, not R&D. New ATS data lacks legacy classifier fields, so DBSCAN is the approved multipath method for the current dataset. A paper, 2019 study, or recent DBSCAN dataset must remain available as a regression and validation reference while code changes proceed. Legacy projects must continue running, work must remain on a feature branch, and `main` must not be modified directly.

## Guidance Incorporated
- Maintain a permanent legacy-study or DBSCAN test and validation dataset.
- Preserve parity with paper/2019 legacy outputs before considering a merge.
- Use DBSCAN for new ATS data where SVM, GaussianNB, CART, and KNN features are unavailable.
- Retain pulse-rate blanking before clustering whenever pulse rate is known.
- Prefer time-windowed or `(time, DDoA)` clustering when SNR/NBW-style signal features are absent.
- Treat HDBSCAN as a possible future upgrade, not current required work.
- Visualize fish distributions using 3D point clouds, voxel density, plan-view heat maps, depth histograms, and kernel density utilization distributions.
- Use 50% kernel contours for core-use area and 95% contours for total-use area where kernel density outputs are requested.
- Work only on feature branch `ENM_jsat3d_edits`; never modify `main` directly.

## Files Updated
- `System_Prompt.txt`
  - Added branch safety and legacy parity rules.
  - Added permanent validation dataset policy.
  - Added DBSCAN multipath policy.
  - Added fish-space visualization policy.
- `LONG_TERM_CONTEXT.md`
  - Added standing decisions for validation, DBSCAN, and branch safety.
  - Added established visualization and clustering methods.
  - Added open limitation: permanent regression dataset remains unselected.

## Decisions & Assumptions
- Decision: DBSCAN is current approved multipath approach for new ATS data.
- Decision: Legacy supervised classifiers are not required for new data when required features are absent.
- Decision: Validation parity is mandatory before merge consideration.
- Decision: Existing legacy code remains protected while new methods are introduced through adapters or isolated modules.
- Assumption: Existing paper, 2019, or recent DBSCAN artifacts can be identified from current repository contents or supplied by manager.

## Blockers & Known Limitations
- Permanent regression dataset has not been selected.
- Exact parity metrics have not been defined.
- New ATS data still lacks SNR and NBW fields.
- FFD3 pulse rate remains unconfirmed.
- HDBSCAN dependency and implementation remain out of scope for current work.

## Next Steps
1. Identify candidate paper, 2019, or recent DBSCAN dataset in repository.
2. Define minimum parity checks against legacy outputs.
3. Separate new DBSCAN path from legacy classifier path.
4. Keep all work on `ENM_jsat3d_edits`.
