# 2026-11-19 - 26E4 position gap 2025-06-28 00:54:59-01:05:47: isolation and census

## Session metadata
- Author: Copilot, AI assistant using Copilot SDK in VS Code.
- Timestamp: 2026-11-19 ~15:30 PST (UTC-08:00) / ~23:30Z.
- Tags: clock-fix, data-loss, 26E4, dbscan, master-clock, census, owner-review.
- Database: `output\11192026_premedian\jsats3d_2025_earlyfish.db` (baseline, opened read-only).
- Method: one-off read-only queries piped to python (no files created; nothing to clean up). No code changed.

## Request
26E4 has no positions between 2025-06-28 00:54:59 and 01:05:47 (study wall clock, PDT shown as +00:00).
Isolate where the data are lost (raw vs processing) and check whether it is widespread.

## Stage trace (26E4, gap vs equal 10.8-min windows before / after)
| Stage | Before | Gap | After |
|---|---|---|---|
| tblDetectionRaw detections (receivers) | 2,946 (19) | 3,259 (19) | 2,421 (19) |
| tblDetectionFilterPrimary / Secondary | 2,946 | 3,259 | 2,421 |
| transmissions (transNo) | 218 | 218 | 219 |
| tblPositions_Deng rows attempted | 10,538 | 13,556 | 4,672 |
| root B 'solution found' | 743 | **23** | 1,184 |
| transmissions with any solution | 145 | **5** | 177 |
| DBSCAN-kept transmissions | 141 | 5 | 154 |
- Raw, multipath and epoching are intact. Loss is at Deng: in the gap 6,486 of 6,778 root B attempts end
  'negative quadratic discriminant - no solution' (inconsistent arrival times).

## Why the arrival times are inconsistent
- Clock knots (`tblDetectionClockFixed`, master beacon 7DB7) have a 10-25 min hole spanning the event on
  almost every receiver, with a correction step across it: about -0.40 s on CFD01-04/07/09, ZOI05/07/09/10
  (common-mode = master), ZOI01 -1.405 s, ZOI02 -1.809 s, ZOI03 +0.193 s. Fish times inside the hole get a
  linear ramp (np.interp) per receiver, so receivers disagree by up to ~1.4 s (~2 km) -> Deng fails.
- Master-beacon epochs inside the hole per receiver: raw 3-26, tblMetronomeFiltered 3-24,
  tblMetronomeSecondFiltered 3-8 (all receivers), **tblDetectionClockFixed 0** (except CFD08 2, master).
  -> removed inside `jsats3d.clock_fix` (jsats3d.py ~1034-1061): 5 iterative DBSCAN passes on (60 s grid time,
  interpolated DDoA) with eps = 90th/95th percentile spacing (~one grid step). A real level step (0.4 s ~ 590 m
  DDoA) makes the interpolated points across it sparse -> noise -> knots deleted; each pass re-interpolates across
  the wider hole and erodes further. Matches 2026-11-03 audit ("post-join DBSCAN rejection").

## Raw-data component (master ZOI08)
- ZOI08 Internal evidence: offset_change/counter_restart at 01:01:00, 01:02:01, 01:03:00, 01:04:01, 01:05:03,
  plus one_second_adjustment rows 01:02:03-07.
- ZOI08's own 7DB7 intervals (normally ~64.0 s; ~63.5 s periodic catch-up at 00:57:09, 01:13:10 is normal):
  62.421 (01:01:24), 64.036, 63.012 (01:03:31), 65.021 (01:04:36), 66.018 (01:05:42) -> the master clock
  stepped several times 01:01-01:06 (net ~+0.4 s, the common-mode step seen on all receivers).
- Conclusion for this event: 00:55-01:01 (~6 min) = processing loss (DBSCAN erosion before the step, master
  stable); 01:01-01:05:41 (~5 min) = genuine master clock instability in raw data, which the pipeline cannot
  represent (master is the time reference; multi-step master resync is not modelled).

## Census - whole run (06-15 to 08-06, all receivers except master)
- Stepped clock gaps (knot gap > 180 s with |correction step| > 50 ms): **1,748 intervals, 17 receivers,
  1,269 receiver-hours**. Largest: ZOI01 264 (171.6 h), ZOI02 214, CFD06 171, CFD08 165, ZOI03 128, CFD03 113.
- Fish detections inside such gaps: **46,642 of 338,114 (13.8%)**.
- Transmissions with >= 1 receiver detection inside a gap vs none, % with a root B solution:
  1D5A 43.6% vs 60.7% (12,631 vs 5,243 transmissions); 9F8C 1.1% vs 27.1%; B5ED 30.3% vs 52.7%;
  26E4 70.3% vs 52.6% (not lower; other receivers often suffice).
- ZOI08 (master) has 410 offset_change rows in the run.
- Caveat: a stepped gap can also be a genuine raw outage; this census does not separate DBSCAN erosion from
  missing data per interval (the 26E4 event shows both).

## Conclusion
- Not a raw-data loss: 26E4 was heard normally. Positions are lost at Deng because clock correction across the
  event is wrong: `clock_fix` DBSCAN erodes clock knots around real clock steps, then fish times are linearly
  interpolated across the hole. Part of this event (01:01-01:05) is a genuine multi-step master resync.
- Widespread: ~14% of fish detections sit in such intervals; transmissions touching them solve markedly less
  often for 3 of 4 tags.

## Options (not implemented; owner decision; clock_fix is shared with 2019 -> any change must be gated to 2025)
1. Stop DBSCAN erosion at steps: detect steps first and run the multipath DBSCAN per segment, or use a fixed
   physically-derived DDoA tolerance instead of percentile eps in mixed (s, m) units; no iterative widening.
2. Never interpolate fish times across a stepped gap: piecewise correction with the step placed at the flagged
   Internal event time (receiver or master), otherwise exclude that receiver's detections inside the gap.
3. Master resyncs: use ZOI08 Internal flags/beacon intervals to correct the master's own steps (ToT) before sync.

## Parameter changes
- None.

## Files touched
- This journal only.
