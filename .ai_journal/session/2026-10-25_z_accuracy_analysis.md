# 2026-10-25: Fish Depth (Z) Accuracy Analysis

## Receiver Depths (from config, with hydrophone offsets applied)

| Receiver Group | Z (m) | Depth (ft) |
|---------------|-------|-----------|
| CFD surface | -3.05 | 10.0 |
| ZOI surface | -2.26 to -3.25 | 7.4 to 10.7 |
| ZOI deep | -10.29 to -12.40 | 33.8 to 40.7 |

## Known Tag Depths (from released_v0.csv)

| Tag | Known Depth |
|-----|------------|
| 1D5A | -1.5 m (5 ft) — near surface |
| 9F8C | -4.57 m (15 ft) — mid-water |
| B5ED | -4.57 m (15 ft) — mid-water |
| 26E4 | Unknown (not in released_v0) |

## Z Accuracy Results

### Unfiltered Root-B (realistic range only, Z ∈ [-30, 0])

| Tag | Known | Measured p50 | Deviation | Realistic % |
|-----|-------|-------------|-----------|------------|
| 1D5A | -1.5 m | **-3.1 m** | **-1.6 m** | 48.6% |
| 9F8C | -4.57 m | **-8.2 m** | **-3.7 m** | 51.9% |
| B5ED | -4.57 m | **-8.3 m** | **-3.7 m** | 66.3% |

### DBSCAN Filtered

| Tag | Known | Measured p50 | Deviation |
|-----|-------|-------------|-----------|
| 1D5A | -1.5 m | -4.7 m | -3.2 m |
| 9F8C | -4.57 m | -12.8 m | -8.2 m |
| B5ED | -4.57 m | -10.6 m | -6.1 m |

### Above-Water Fraction (unfiltered Root-B)

| Tag | Above Water | Extreme Negative (< -30m) |
|-----|------------|--------------------------|
| 26E4 | 27.4% | 0.7% |
| 9F8C | 45.9% | 2.2% |
| 1D5A | 50.6% | 0.8% |
| B5ED | 30.7% | 3.0% |

## Assessment

### What's Good
- **1D5A unfiltered is close**: p50 = -3.1 m vs known -1.5 m. Only 1.6 m too deep. Given the array geometry and clock residuals, this is **reasonable**.
- **The time-dependent Z pattern is visible**: Fish show depth changes over time (e.g., 9F8C starts deeper and comes shallower), which is biologically plausible behavior.
- **Root B is clearly better than Root A**: Root A medians are -17 to -45 m (way too deep), while Root B is -3 to -8 m.

### What's Wrong
- **Systematic negative bias**: All three tagged fish are measured deeper than their known depths. The bias is ~3.7 m for 9F8C and B5ED, ~1.6 m for 1D5A.
- **This is a pipeline issue, not a fish behavior issue**: The tags were physically deployed at known depths. The pipeline consistently measures them deeper.
- **DBSCAN makes it worse**: The DBSCAN filter preferentially keeps deeper solutions (which have better geometry from more receiver sets), pushing the median even deeper. For 9F8C, DBSCAN makes the deviation go from -3.7 m to -8.2 m.
- **~30-50% of solutions are above water**: This is the same Root-B outlier problem we saw in the ENT drag tests. The clock correction fixed the worst of it (vs. the pre-fix ~100% above water), but a significant fraction remains.

### Likely Causes of the Z Bias

1. **Sound speed table calibration**: The pipeline uses a freshwater sound-speed table based on temperature. If the actual sound speed is ~1-2 m/s different from the table value, it would systematically bias all range estimates (and thus Z) by ~1-2%. At the array scale (~100 m), this could explain 1-2 m of bias.

2. **Residual clock corrections**: ZOI08 (master beacon) is unsurveyed, carrying residual bias into all clock corrections. CFD01/CFD09/ZOI11 have no correction curves at all (~940 ms late). These uncorrected receivers contribute to Z bias.

3. **ZOI08 position error**: The ZOI08 beacon host offset (+0.56, +1.06 m) was applied, but ZOI08 itself is unsurveyed. Any error in ZOI08's position propagates into all clock corrections and thus all fish positions.

4. **Deng solver geometry**: The flat array (9 m vertical spread between surface and deep receivers) creates a geometry where Z solutions are inherently less constrained than XY. The solver may prefer deeper solutions when geometry is marginal.

### Recommendation

**For client delivery, use unfiltered Root-B with a Z sanity filter (|Z| < 30 m).** The DBSCAN filter makes the Z bias worse. The unfiltered realistic range gives the closest match to known tag depths.

For the systematic bias (~3.7 m for 15 ft tags), this is within the expected accuracy floor given:
- The 3.95 m horizontal accuracy we achieved on ENT drag tests
- The unsurveyed ZOI08 and uncorrected CFD01/CFD09/ZOI11 receivers
- The sound speed table calibration uncertainty

The plots look **visually reasonable** — the fish tracks show coherent movement patterns, the Z color gradients show plausible depth changes, and the 3D plots show fish in the water column between the surface and deep receivers. The absolute Z values are off by ~2-4 m, which is acceptable for behavioral analysis but should be noted.
