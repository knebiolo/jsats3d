# jsats3d Analysis Process (flowchart reference)

Operation-level description of the jsats3d analysis pipeline for building the
study flowchart. Each numbered line is one flowchart box; {curly braces} mark
decision points. Technique names are kept; code, variable, and table names are
not.

## A. Project setup

1. Create project database
2. Enter study settings: units, study time window, reference receiver
3. Load tag list (ping rates), surveyed receiver positions, water level,
   water temperature, optional bathymetry

## B. Raw data import

4. Parse receiver files into microsecond-precision timestamps
5. Clip to study window; drop duplicates
6. Pool all receivers' detections

## C. Beacon epoch assignment

7. Number pings at each beacon's own receiver (time gap >= 1/2 ping
   interval = next ping)
8. Assign detections at all other receivers to the shared epochs (match
   within +/- 1/2 ping interval)

## D. Beacon multipath removal

9. Preliminary first-detection filter: first arrival per epoch per receiver =
   direct signal; later arrivals = multipath
10. Train classifier (SVM / Naive Bayes / CART / KNN) on signal metrics
    (amplitude, noise bandwidth, SNR)
    {insufficient training data? -> unsupervised DBSCAN instead}
11. Re-screen all beacon detections with the classifier

## E. Clock synchronization (per receiver)

12. Calculate TDOA for every shared epoch: arrival time at this receiver -
    transmit time at reference receiver
13. Calculate expected TDOA from receiver spacing / speed of sound (from
    water temperature)
14. Residual = measured - expected TDOA -> clock-error time series
15. Cluster with DBSCAN (iterated 5x) to strip remaining multipath/outliers
    from the error series
16. Detect clock jumps (step changes in the error series); split the series
    at each jump
17. Fit piecewise linear regression to the clock-error series (one piece per
    segment)
18. Apply correction to all data from that receiver:
    corrected time = recorded time - predicted clock error

## F. Locate submerged receivers (two-pass loop)

Note: steps C-E (pass 1) run with only the surveyed surface receivers;
deep receivers cannot be surveyed directly, so their positions are estimated
from their own beacons before pass 2.

19. Each deep receiver (e.g., ZOI01-ZOI03) carries its own beacon; treat that
    beacon like a fish tag: multipath-filter its detections (as in D)
20. Solve 3D positions for the beacon using only the surveyed, clock-corrected
    surface receivers (same TDOA math as section H)
21. Take the median of the solutions as the receiver's coordinates; where the
    depth was tape-measured, fix Z and solve only X/Y (3-receiver,
    fixed-depth solution)
22. Loop back (pass 2): discard derived results and re-run epoch assignment,
    multipath removal, and clock synchronization (C-E) with the deep
    receivers now included

## G. Fish detection cleanup

23. Apply the per-receiver clock corrections to fish detections
24. Assign fish detections to epochs using the tag's ping interval
25. Remove multipath: first-detection filter -> classifier re-screen
    (same as D)

## H. 3D positioning (per fish tag)

26. Gather the tag's cleaned detections: only pings that survived both
    multipath filters, only from clock-corrected receivers
27. Build the receiver geometry:
    - surveyed X/Y/Z for fixed receivers
    - mobile receivers: interpolate GPS track to get X/Y at each detection time
    - floating receivers: adjust depth with water level at each detection time
    - construct the 3D convex hull of the array (the "trusted volume")
28. Group detections by ping (shared epoch number); process one ping at a time
29. {>= 4 receivers heard this ping?} -> no: record "not enough receivers",
    skip -> yes: continue
30. Enumerate every unique combination of 4 receivers that heard the ping
31. For each combination: receiver with the earliest arrival = reference;
    compute 3 TDOAs (each other arrival - reference arrival)
32. Look up water temperature at the arrival time -> speed of sound
33. Build the geometry matrix (3D offsets of the other 3 receivers from the
    reference) and solve the exact TDOA equations (Deng 2011) - a quadratic
    in the transmit time, giving two candidate solutions (root A and root B)
34. Validity checks per root - reject with a reason code if:
    - no real solution (negative discriminant)
    - degenerate receiver geometry (singular matrix)
    - negative travel time (ping would arrive before it was sent)
35. Convert each valid root to an X/Y/Z position (offset from the reference
    receiver); flag whether it falls inside the receiver convex hull
36. Physical-plausibility filters (applied before the median consensus, so bad
    candidates cannot bias the result):
    - above the water surface at that moment -> rejected
    - more than ~4 m below the deepest receiver -> rejected
    - outside the array bounding box (footprint + small margin) -> rejected
      (catches clock-glitch artifacts that land far away)
37. Keep all surviving candidates - typically many per ping (one A and one B
    per 4-receiver combination)
38. Alternate branch: if the fish's depth can be assumed, a 2D solution needs
    only 3 receivers (same math with depth fixed)

## I. Track filtering & output

39. Take the median of the surviving candidates per ping (robust to remaining
    outliers) -> one consensus position per ping
40. Hampel smoothing: compare each position to a rolling median of the track;
    reject points that jump more than ~10 m from the local track; lightly
    smooth the rest (rolling median); split the track at time gaps (~20 s)
41. Constant-velocity Kalman filter per track segment (X/Y/Z): blend each
    measurement with predicted motion -> final smoothed track (the product
    used in figures and delivered position CSVs, alongside the raw consensus)
42. Accuracy check: run fixed beacons and GPS-truthed drag tests through the
    same chain; compare to surveyed/GPS locations
43. Optional bathymetry screen: reject stored positions above the water
    surface or below the riverbed
44. Kernel density estimate of space use (3D volumes, day vs. night)

## Side inputs (connectors)

- Water temperature -> steps 13, 32 (speed of sound)
- Water level -> steps 13, 27, 36 (receiver depths, water-surface check)
- Bathymetry -> step 43 (below-riverbed check)
