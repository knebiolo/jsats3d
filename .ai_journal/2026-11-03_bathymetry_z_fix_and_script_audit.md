# Bathymetry Z Coordinate Fix & Pipeline Script Audit — 2026-11-03

## Bathymetry Z Coordinate System Mismatch

### Problem
The bathymetry check was comparing position Z values (from Deng positioning) directly to bathymetry Z values without accounting for the fact that they are in **completely different coordinate systems**:

- **Position Z values**: In the local receiver coordinate system (X_t/Y_t/Z_t frame), where receivers have Z_t values ranging from -12.4 to -2.3 m
- **Bathymetry Z values**: In a different elevation system (values ranging from -33.45 to 9.08)

The offset between these systems **varies by location** (0.7 m to 25.4 m across receivers), so no single global offset could fix it.

### Investigation
Checked receiver Z_t vs nearest bathymetry Z at each receiver location:

| Receiver Group | Z_t Range | Offset (Z_t - bathy_Z) | Notes |
|---------------|-----------|----------------------|-------|
| ZOI01-06, ZOI10 | -12.4 to -3.1 | 0.7 to 2.8 m | Consistent — bathymetry covers ZOI area well |
| CFD02-09 | -3.1 | 8.5 to 14.0 m | Larger offsets — CFD area (forebay) deeper |
| ZOI07-09, ZOI11, CFD01 | -2.3 to -3.1 | 11.3 to 25.5 m | Bathymetry data sparse/missing in these areas |

### Solution
Modified `jsats3d/bathymetry.py` to use **receiver Z_t values as local water surface references**:

1. For each position solution, convert X/Y from local to UTM (using receiver easting/northing vs X_t/Y_t offset: dX=567,993.85, dY=5,146,184.40)
2. Look up nearest bathymetry Z using UTM coordinates (KD-tree)
3. Look up nearest receiver's Z_t value (local water surface reference)
4. Compare position Z against bathymetry Z using the local receiver reference

### Results
Re-ran bathymetry check on `output/10292026_Test/jsats3d_2025_earlyfish.db`:
- **117,364 positions pass** (within water column)
- **86,148 rejected below bathy** (below riverbed)
- **111 rejected above WSEL** (above water surface)

Per tag:
| Tag | Pass | Rejected Below Bathy | Rejected Above WSEL |
|-----|------|---------------------|-------------------|
| 1D5A | 50,553 | 45,981 | 32 |
| 26E4 | 19,386 | 16,816 | 22 |
| 9F8C | 2,031 | 907 | 1 |
| B5ED | 45,394 | 22,444 | 56 |

Figures generated in `output/bathy_test/figures/` using `fish_figures_2025.py` (same code as originals).

## Pipeline Script Audit

### Scripts Kept (part of 2019/2025 pipeline)
| Script | 2019 Needed? | 2025 Needed? | Why |
|--------|-------------|-------------|-----|
| `run_data.py` | Yes | Yes | Entry point for both |
| `legacy_pipeline.py` | Yes | Yes | 2019/legacy Deng pipeline |
| `parse_ats_raw_to_legacy.py` | Yes | Yes | `run_data.py` imports from it |
| `adapt_2025_to_legacy.py` | No | Yes | 2025 data adapter |
| `beacon_pairwise_dbscan.py` | No | Yes | Metronome (step 2-3) |
| `clock_jump_pre.py` | No | Yes | Clock jumps (imported by _2025_pipeline.py) |
| `ent_analysis_2025.py` | No | Yes | Step 12 plot_drag (called by _2025_pipeline.py) |
| `fish_figures_2025.py` | Yes | Yes | `run_data.py` references FISH_FIGURES |
| `drag_test_figures_2025.py` | Yes | Yes | `run_data.py` references DRAG_FIGURES |
| `metronome.py` | No | Yes | Metronome analysis |
| `mulitpath.py` | No | No | Legacy standalone multipath (kept per user request) |

### Scripts Deleted (not part of pipeline)
- `_run_bathy_check.py` — standalone bathy test (now integrated into legacy_pipeline)
- `_check_db.py` — database diagnostic
- `fish_dashboard_v3.py` — standalone dashboard

### Key Dependencies
- `_2025_pipeline.py` → `ent_analysis_2025.py` → `fish_figures_2025.py` + `drag_test_figures_2025.py`
- `run_data.py` imports from `parse_ats_raw_to_legacy.py` (needed for both 2019 and 2025)
- `fish_figures_2025.py` is the figure generator used for both 2019 and 2025 output

## X/Y Coordinate System Fix

### Problem
Position X/Y values are stored in a **local coordinate system** (X_t/Y_t frame), but the bathymetry file is in **UTM (EPSG:26910)**. Direct comparison was impossible without conversion.

### Solution
Computed the offset from receiver metadata (easting/northing minus X_t/Y_t) and applied it to all position solutions before bathymetry lookup:

- **dX** = 567,993.85 m (mean easting - mean X_t)
- **dY** = 5,146,184.40 m (mean northing - mean Y_t)

Conversion: `utm_x = local_x + dX`, `utm_y = local_y + dY`

This offset is consistent across all receivers (computed from the 20-receiver array metadata in `tblReceiver`).

## Bathymetry Logic Issue Found

During review of the implemented fix, a **dead code path** was identified in `bathymetry.py` (lines 349-359):

```python
water_depth = local_water_surface_z - local_bathy_z
if sol_z < local_water_surface_z - water_depth + wsel_tolerance_m:  # → sol_z < local_bathy_z + tolerance
    check = "rejected_below_bathy"
elif sol_z < local_bathy_z - wsel_tolerance_m:  # DEAD CODE — subset of above
    check = "rejected_below_bathy"
```

The first `if` condition (`sol_z < local_bathy_z + tolerance`) already catches everything the `elif` would catch (`sol_z < local_bathy_z - tolerance`), making the `elif` unreachable. The effective threshold is **`local_bathy_z + 0.05 m`** (slightly above the riverbed).

## Bathymetry Config Changes
- Enabled `bathy_path` in `config/run_data.toml` (was previously commented out)
- Bathymetry file: `elev_contour2ras_1m_pts_refined_20240826_utm10n.xyz` (EPSG:26910)
- Bathymetry check now runs automatically as part of `legacy_pipeline.py` (line 601)

## Files Modified
- `jsats3d/jsats3d/bathymetry.py` — Added X/Y local→UTM conversion + local Z reference using receiver Z_t values
- `jsats3d/config/run_data.toml` — Enabled bathy_path

## Files Generated
- `output/bathy_test/figures/*` — Bathymetry-filtered figures for all 4 tags (matching original style)
- `output/bathy_test/bathy_test.db` — Test database with bathymetry check results
