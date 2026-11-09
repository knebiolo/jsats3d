# 2026-11-04: A/B Validation of Clock-Jump Corrections (incl. TDOA-DBSCAN Supplement) — 2025 Early-Fish

**Date:** 2026-11-04 (journal written 2026-11-05)  
**Author:** AI agent  
**Purpose:** PI requested a full Arm A vs Arm B validation (fish positions + clock corrections) showing that the clock-jump corrections — including the new TDOA-DBSCAN supplement — do what they are supposed to, without degrading results.

---

## PI Request

1. Show Arm A vs Arm B TDOA/clock comparisons for the 2025 early-fish data.
2. Pick a few random jumps, find fish in the forebay at those times, and plot the fish tracks — Arms A vs B — for 2–3 example fish.
3. Confirm the corrections are not degrading results (rescued / broke / no-impact, per the 2026-10-29 A/B scheme).
4. Journal the work and findings; get rid of the one-off diagnostic scripts when done.

**Arms:**
- **Arm A** = `output\10292026_Test` — unshifted baseline; jumps handled only by in-sync auto-detect.
- **Arm B** = `output\11032026_clocksync` — same pipeline with `clock_tdoa_supplement = true`: 696 internal-code jumps (stage 1) + 1653 TDOA-DBSCAN supplement jumps (stage 2: 1604 receiver, 49 master), applied BEFORE the piecewise-regression clock sync. Total applied: **2349** (|offset| median 1.000 s, p99 3.619 s, max 4.622 s). Candidates: 2598 = 1519 accepted + 1079 unresolved (830 applied at an uncertain boundary + 249 residual/ambiguous steps still uncorrected).

---

## A/B Design

The production Arm B run is **gated at the clock review** (`clock_review_gate.csv`, 630 issues, status `blocked`), so its fish steps never ran in production. To get Arm B fish positions:

1. **Throwaway copy** of the Arm B DB (full 5.26 GB compact DB) at `output\ab_B2_2026-11-04\`, with a rewritten `.compact.json` manifest (target = copy path; `build_context` requires status complete + target match) and a temp `run_file.toml` = the 11-03 clocksync toml with `output_db` retargeted.
2. **`tblClockExperimentPolicy` dropped on the copy only** (after `build_context`), because (a) the strict-support gate in `jsats3d.py` `multipath_data_object` (~line 336) otherwise raises "Clock review required: %d unsupported detections…", and (b) Arm A's DB has no policy table, so dropping it makes both arms run the **identical legacy `np.interp` fish code path** (the gate only refuses unsupported rows; the arithmetic is the same `np.interp`).
3. Ran the pipeline's **own step functions** (`fish_multipath_run`, `positions_run`, `export_run`, `positions_dbscan_run` in `_2025_pipeline.py`) directly on the copy — the gate only exists in the main loop (lines 1513–1517), so direct calls bypass it. Steps: fish_multipath 0.8 min, positions 1.6 min, export 0.1 min, positions_dbscan 0.1 min. Root-B solutions found: 26E4 33,155 / 9F8C 2,642 / 1D5A 67,429 / B5ED 33,869.
4. Arm B client CSVs + figures generated with production `fish_figures_2025.py --depth-floor-m -16.0`, matching Arm A's depth floor.

**The production DB, its candidate ledger, and the gate are untouched.** This comparison is a diagnostic, not an approval.

**Comparison scheme (matches `output\jump_ab_report_2026-10-29\REPORT.md` for comparability):**
- Array center = receivers_3d centroid **(467.0, 113.3) m**; validity radius **2000 m**; affected window **±300 s** of any applied jump; cross-arm transmission pairing **by time, nearest pair within 1 s** (NOT by transNo — see Lessons L2).
- **Raw level** (the Oct-29 verdict): per-transmission median of root-B Deng solutions from `tblPositions_Deng` (`comment='solution found' and solution='B'`) in each arm's DB; "valid" = within 2 km of the centroid.
- **Consensus level** (final client product, `*_positions.csv` consensus rows): the consensus chain's 1 km gate makes every consensus fix in-array in BOTH arms by construction, so this level reports **coverage and agreement**, not validity.
- **Curve re-check**: residual TDoA step measured in each arm's saved clock curve at every applied jump time (top 4 receivers by applied-jump count: ZOI01 371, ZOI02 257, CFD06 227, ZOI03 226; Arm B evaluated in corrected time via the ledger's `corrected_before_s`/`corrected_after_s`; ≥3 beacon epochs per side).

---

## Findings

### 1. Saved clock curves — the jumps are really gone in Arm B (strongest evidence)

| arm | measurable jumps | median \|step\| | p99 | max | ≥ 50 ms |
|---|---|---|---|---|---|
| A (unshifted) | 2 | 415.6 ms | 425.2 ms | 425.4 ms | 2 |
| B (supplement) | 463 | 0.013 ms | 0.133 ms | 0.256 ms | 0 |

Arm A's curve gaps at most of its own auto-detected jumps, so only 2 of its jump times are measurable — and both show the full ~420 ms jump still present. Arm B's curve is continuous across **all 463 measurable applied jumps** (max residual step 0.256 ms). Same pattern as the 2026-10-29 report (4 measurable, 392–406 ms).

### 2. Raw root-B solutions (the 2026-10-29 verdict)

| Set | n | both valid | rescued (A bogus → B valid) | broke (A valid → B bogus) | both bogus | valid % A → B |
|---|---|---|---|---|---|---|
| All matched | 12,651 | 12,625 | **6** | **8** | 12 | 99.86% → 99.84% |
| Affected (±300 s of a jump) | 3,130 | 3,123 | **2** | **1** | 4 | 99.81% → 99.84% |
| No-impact | 9,521 | 9,502 | 4 | **7** | 8 | 99.87% → 99.84% |

- **Rescued (6)** — dramatic gross-error fixes, e.g. 26E4 2025-06-27 16:47:49: Arm A 548.7 km from array center → Arm B 20.3 m (the same 548 km transmission class as Oct 29). Others: 26E4 ×2 (2.1–2.4 km → 20–23 m), 1D5A (29.2 km → 35 m), B5ED (5.3 km → 65 m).
- **Broke (8)** — 6 of 8 fall **inside documented unresolved windows** from the candidate ledger (ZOI06 uncertain ×5, CFD06 uncertain ×2, with overlap); the other 2 sit adjacent (nearest gap 1364 s to a ZOI06 uncertain window, 180 s to a CFD07 residual window). In every case inspected, the degraded Arm B solutions are the 4-receiver combinations carrying a receiver with an **uncertain-boundary correction (chiefly ZOI06)** — the expected residual effect of corrections applied at approximate boundaries, i.e. exactly the set the production gate flags. None falls in a clean region away from the unresolved set.
- Coverage: raw transmissions A 14,611 → B 16,733 (**+2,122**; B-only 4,082 vs A-only 1,960).

### 3. Final consensus product (client CSVs)

| tag | consensus A | consensus B | matched | A-only | B-only |
|---|---|---|---|---|---|
| 26E4 | 1,398 | 1,774 | 1,090 | 308 | 684 |
| 9F8C | 76 | 134 | 28 | 48 | 106 |
| 1D5A | 3,465 | 4,231 | 3,409 | 56 | 822 |
| B5ED | 2,473 | 2,337 | 1,344 | 1,129 | 993 |
| **total** | **7,412** | **8,476** | **5,871** | **1,541** | **2,605** |

- Arm B nets **1,064 more final fixes (+14.4%)**. The coverage difference is spread across the tracked spans, not localized at jumps: A-only is 22.6% within ±300 s of a jump vs 21.8% of ALL Arm A fixes (no enrichment); B-only is 36.9% vs 26.3% of all Arm B fixes (enriched around jumps — where Arm A's chain dropped transmissions). It is a re-solving effect of two different clock models, not a position-quality change.

### 4. No-impact degradation (matched consensus transmissions outside all jump windows)

n = 4,601: **p50 = 8.3 cm, p90 = 4.3 m, p99 = 16.3 m, max = 37.7 m**; 48.4% bit-identical (<1 cm), 63.8% <1 m, **97.4% <10 m**. By distance band the deltas do not grow with distance to the nearest jump (300–600 s: p50 4 mm; 600–1800 s: p50 3 mm; 1800–3600 s: p50 2.4 m; 3600–7200 s: p50 2.6 m). No degradation signal.

### 5. Example fish (PI's explicit ask)

Three distinct fish, each isolated around an applied jump, Arms A vs B side by side (`output\jump_ab_report_2026-11-04\example_fish_*.png`, ±30 min window, jump at center):

- **1D5A** around the **ZOI07** jump −1.022 s at 2025-06-18 17:01:05 local [tdoa_dbscan_receiver]: 15 A / 33 B consensus fixes in ±300 s; max A-B delta 13 m.
- **26E4** around the **ZOI01** jump +1.006 s at 2025-06-28 00:41:00 local [tdoa_dbscan_receiver]: 123 A / 131 B; max delta 12 m.
- **B5ED** around the **ZOI05** jump +1.005 s at 2025-06-26 18:01:08 local [OffsetChanged]: 47 A / 50 B; max delta 11 m.

Tracks agree closely in both arms (both stay in-array — the consensus product is 100% in-array by construction); Arm B shows equal-or-better coverage and a modest (~10 m) position shift around the jump. Note: a "bogus-in-A, valid-in-B rescue" at the *consensus* level does not occur for these tags — rescues live at the raw level (Finding 2), because the consensus chain's 1 km gate already drops gross errors in both arms.

Plus 18 `tdoa_arms_*.png` TDOA-level arm plots (one per receiver, fish-covered interval) from the production clock_review step in `output\11032026_clocksync\clock_review\interval_*`.

---

## Caveats / Scope

- **Diagnostic only**: Arm B fish steps ran on a throwaway copy with the strict clock policy table dropped (both arms then use the identical legacy fish code path). Nothing here approves the 830 uncertain-boundary or 249 residual corrections; **the production gate is unchanged and still blocks fish processing (630 issues: 519 boundary-uncertain, 92 overlapping-regime, 19 residual-step; receivers ZOI01–03, 05–08)**.
- The 8 broke cases are the visible cost of the 830 approximate-boundary corrections + 249 uncorrected residuals; resolving the gate's issue list is the path to removing them.
- Consensus-level "validity" (2 km radius) is degenerate by construction at this product level — the raw root-B level is the honest rescued/broke comparison (same as Oct 29).
- CFD05 (no clock data), ZOI04, ZOI11 are not in the 3D receiver set; their detections never enter these fixes. ZOI04/ZOI11 also have no clock curves in either arm (harmless warning during fish_multipath).
- Arms use their own receiver-time coordinates; only transmission-level position deltas are compared, never raw timestamps across arms.

## Next Steps (open, owner/PI-driven)

1. **Resolve the 830 uncertain boundaries + 249 residuals** (the gate's issue list). PI's idea of a tdiff → 3-group DBSCAN pass (separating direct / steady-reflection / anchor-side groups in the regime gap) is **not yet implemented** — this run used regime-level DBSCAN. That is the expected path to clearing the 519 boundary-uncertain issues.
2. **Validation-drag period analysis** (quantifying how much the tag-drag period drags validation) not done.
3. Once the gate clears: re-run the fish steps on the **production** DB (with the policy table), regenerate client deliverables, and retire the copy.

---

## Files

| Path | Status |
|---|---|
| `output\jump_ab_report_2026-11-04\` | **Kept** — REPORT.md, summary.json, per_tag_summary.csv, raw_level_matched.csv, raw_broke_cases.csv, affected_windows.csv, noimpact_check.csv, decay_by_distance_to_jump.csv, curve_steps_at_jumps.csv, fish_steps_on_copy.json, example_fish_*.png, drift_*.png |
| `output\ab_B2_2026-11-04\` | **Kept** — throwaway Arm B copy (DB, .compact.json, run_file.toml, figures with Arm B `*_positions.csv`) |
| `output\10292026_Test\`, `output\11032026_clocksync\` | **Untouched** — production Arm A and Arm B outputs incl. clock_review ledger/gate |
| `scripts\_ab_probe.py`, `_ab_check_policy.py`, `_ab_check_curves.py`, `_ab_debug_picks.py`, `_ab_validate.py`, `_ab_analyze.py`, `_ab_debug_broke.py`, `_ab_debug_b5ed.py`, `_ab_debug_broke2.py` | **Deleted** after report generation (project precedent: Oct 29's one-off script was deleted after its report) |

## Lessons

- **L1 — pandas parses an all-`26E4` tag column as float64.** `"26E4"` is a valid float literal (2.6e4 = 260000.0), so `pd.read_csv` infers `float64` for a tag column containing only that tag, and `df.tag == "26E4"` silently matches nothing. Load client CSVs with `dtype={"tag": str}`. (Cost: 26E4 — 1,398/1,774 consensus fixes — was invisible to the per-tag comparison for most of this session.)
- **L2 — cross-arm transmission matching must be by time, not transNo.** The applied jumps shift receiver time, so transNo numbering only agrees between arms inside individual clock segments (26E4's whole 2.1-h span sits in one shifted segment → 0 transNo matches). Match by nearest pair within 1 s (the Oct-29 scheme).
- **L3 — pandas index alignment bites with `.loc` selections.** `a.loc[ia, col] - b.loc[ib, col]` aligns on index *labels*, not positions; with disjoint label sets the result is all-NaN. Use `.to_numpy()` before arithmetic on two separately selected frames.
- **L4 — validity radius on the consensus product is degenerate.** The consensus chain's 1 km gate guarantees every consensus fix is in-array in both arms, so a 2 km validity test there is uninformative; run rescued/broke on the raw root-B per-transmission medians.
