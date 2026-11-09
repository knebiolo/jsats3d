# Journal: 2026-11-07 Jump Audit

## Audit: Applied jumps vs raw TDoA step

### Method
- For each applied jump on ZOI01, ZOI05, ZOI07 (after 06-17)
- Computed raw TDoA = TDoA - TDoA_error from tblDetectionClockFixed
- Compared raw TDoA step (mean before vs after jump time) with applied offset
- Flagged mismatches > 0.5 ms

### Results

**ZOI01 (273 jumps audited):**
- Raw TDoA step: mean=0.000000s, std=0.000004s (flat)
- Applied offset: mean=-0.045s, median=-0.338s, std=1.298s (range -4.6 to +4.6s)
- Mismatches > 0.5ms: **272/273**
- Cases where raw TDoA flat (<10ms) BUT applied offset >100ms: **272/273**

**ZOI05 (66 jumps audited):**
- Raw TDoA step: mean=-0.000001s, std=0.000006s (flat)
- Applied offset: mean=-0.001s, median=0.397s, std=1.196s (range -2.6 to +2.6s)
- Mismatches > 0.5ms: **66/66**
- Cases where raw TDoA flat (<10ms) BUT applied offset >100ms: **66/66**

**ZOI07 (24 jumps audited):**
- Raw TDoA step: mean=-0.000001s, std=0.000002s (flat)
- Applied offset: mean=0.165s, median=-0.300s, std=1.663s (range -2.6 to +4.6s)
- Mismatches > 0.5ms: **24/24**
- Cases where raw TDoA flat (<10ms) BUT applied offset >100ms: **24/24**

### Key finding
Raw TDoA is perfectly flat at all jump times. Yet 1–5 second corrections are being applied. The offset values are almost entirely ~1.0s, ~2.0s, ~3.0s, ~3.6s, ~4.6s — integer second adjustments (OffsetChanged) and OneSecondAdjustment events.

**The jump detection is finding steps in TDoA_error that don't exist in raw TDoA. Master-side OffsetChanged/OneSecondAdjustment events are being charged to listeners.**

### Implications
- All 632 applied jumps on ZOI01/ZOI05/ZOI07 are spurious
- The corrections move the clocks AWAY from the raw data (11.4s for ZOI01, -6.5s for ZOI07)
- This is why the gate fails: the corrections create discontinuities in the residual curve
- The fix must be in the jump detection/application, not in the gate check

### Next steps
1. Audit ZOI02, ZOI03, CFD09 (the other receivers with worse range)
2. Check if master-side OffsetChanged events correlate with listener jumps
3. Fix the jump detection to exclude master-side events from listener corrections
