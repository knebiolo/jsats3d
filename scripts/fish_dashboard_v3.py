"""Dashboard v3 — filterable interactive HTML for the earlyfish 2025 study.

Built for fisheries biologists / hydropower PMs. One self-contained HTML file:
  - fish roster (tag, species, release, recapture, fate)
  - combinable filters: species, tag, time window, water-column temperature
  - 3D movement view (one tag by default, "all tags" mode, clean labels)
  - water-column temperature + WSEL panels with fish observation windows
  - depth-vs-time view coloured by local water temperature

Data sources
  - output/jsats3d_2025_earlyfish.db (read-only): receivers, root-B solutions, WSEL
  - K: .../Temperature/2025_Temp_String_Data_5min_interpolated.csv (5-min water column)
  - K: .../3_PTAGIS_datasets/released_v0.csv, collected_v0.csv (species, release, recapture)

Time basis
  The study time basis is PDT (UTC-7) wall clock (journals 2026-09-24/25; the 06-10
  drag test is planned 09:00-13:00 PDT and its GPS-anchored figure titles read
  10:24-10:54 on the same epoch axis). Position epochs (ToA - T01) and the
  temperature/WSEL timestamp strings are all PDT wall clock. This dashboard displays
  everything in local (PDT) time; UTC = local + 7 h (shown in hover).

Track chain (verified item12, identical to fish_figures_2025.py):
  root-B solutions -> plausibility gate [floor, 0] -> receiver-set consensus
  -> Hampel smooth -> Kalman.  Self-check: Kalman point counts must match the
  verified fish_qc.csv (26E4 1387, 9F8C 65, 1D5A 3413, B5ED 2327).

Usage:
    python scripts/fish_dashboard_v3.py [--db DB] [--out DIR]
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import base64
import json
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ent_analysis_2025 import legacy_receiver_set_consensus  # noqa: E402
from drag_test_figures_2025 import (smooth_track, kalman_filter_track,  # noqa: E402
                                    resolve_depth_floor, run_file_from_status)

REPO = HERE.parent
DEFAULT_DB = REPO / "output" / "jsats3d_2025_earlyfish.db"
DEFAULT_OUT = REPO / "output" / "pm_figures_2026-10-22" / "fish_dashboard_v3"
K_ROOT = Path(r"K:\Jobs\5662\001\Data\DataTrans\2025_Data")
TEMP_CSV = K_ROOT / "Temperature" / "2025_Temp_String_Data_5min_interpolated.csv"
PTAGIS_DIR = K_ROOT / "CowlitzAT2025_Data_Deliverables" / "3_PTAGIS_datasets"

STUDY_TAGS = ("26E4", "9F8C", "1D5A", "B5ED")
TAG_COLORS = {"26E4": "#1f77b4", "9F8C": "#d62728", "1D5A": "#2ca02c", "B5ED": "#9467bd"}
FATE_COLORS = {"PASSED": "#27ae60", "REACHED DAM": "#16a085",
               "FOREBAY-LOST": "#e74c3c", "UNKNOWN": "#f39c12"}
# Verified Kalman point counts (fish_qc.csv, 2026-10-25) — self-check contract.
EXPECTED_KALMAN = {"26E4": 1387, "9F8C": 65, "1D5A": 3413, "B5ED": 2327}

# Zone proxies (same as the delivered v2 dashboard; documented as proxies).
DAM_FACE_Y = 75.0
FOREBAY_Y = 90.0
TEMP_STATIONS = {  # station -> depth sensor columns (m encoded in column name)
    "DD_N": ["DD_N_0p5", "DD_N_1p5", "DD_N_9", "DD_N_18"],
    "FBS": ["FBS_0p5", "FBS_1p5", "FBS_6", "FBS_9", "FBS_15"],
    "PDS": ["PDS_0p5", "PDS_1p5", "PDS_3", "PDS_6"],
    "DD_S": ["DD_S_0p5", "DD_S_1p5", "DD_S_9", "DD_S_12", "DD_S_21"],
}
TEMP_REF_DEPTH = "DD_N"  # station used for per-point local temperature
TEMP_DISPLAY_DOWNSAMPLE = 3  # 5-min -> 15-min for embedded display series


def col_depth_m(col):
    """'DD_N_0p5' -> 0.5"""
    return float(col.rsplit("_", 1)[-1].replace("p", "."))


# --------------------------------------------------------------------------- loading
def wallclock_epoch(series):
    """PDT wall-clock datetimes -> epoch seconds on the study basis (wall clock stored
    as if UTC, exactly like the position epochs). Date-only values become 00:00:00."""
    s = series.astype(str).str.strip()
    s = s.mask(s.str.match(r"^\d{4}-\d{2}-\d{2}$"), s + " 00:00:00")
    dt = pd.to_datetime(s, errors="coerce")
    # Unit-safe conversion: to_datetime may return ns OR us resolution depending on
    # input format/pandas version, so normalize to ms explicitly before taking the
    # int64 view (a raw astype("int64") // 1e9 silently corrupts us-resolution data).
    ms = dt.to_numpy().astype("datetime64[ms]").astype(np.int64)
    out = (ms / 1000.0).astype(np.float64)
    out[~dt.notna().to_numpy()] = np.nan
    return out


def b64f32(arr):
    return base64.b64encode(np.asarray(arr, dtype=np.float32).tobytes()).decode()


def b64f64(arr):
    return base64.b64encode(np.asarray(arr, dtype=np.float64).tobytes()).decode()


def load_temp_watercolumn():
    """5-min water-column temperatures -> {station: {col: (t0, dt_s, vals f32)}}."""
    df = pd.read_csv(TEMP_CSV)
    t0 = wallclock_epoch(df["DateTime"])
    keep = np.isfinite(t0)
    df, t0 = df.loc[keep], t0[keep]
    order = np.argsort(t0, kind="stable")
    df, t0 = df.iloc[order].reset_index(drop=True), t0[order]
    dt = float(np.median(np.diff(t0)))
    out = {}
    for st, cols in TEMP_STATIONS.items():
        out[st] = {c: (float(t0[0]), dt, df[c].to_numpy(dtype=np.float32)) for c in cols}
    return out


def load_ptagis_metadata():
    """acoustic tag (upper) -> metadata dict; recaptures keyed by 3DD code."""
    meta, recaptures = {}, {}
    if not PTAGIS_DIR.exists():
        return meta, recaptures
    rel = pd.read_csv(PTAGIS_DIR / "released_v0.csv", dtype=str)
    rel["at"] = rel["Acoustic Tag Value"].fillna("").str.strip().str.upper()

    def _s(v):
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()

    for _, r in rel[rel["at"].str.match(r"^[0-9A-F]{4}$")].iterrows():
        meta[r["at"]] = dict(
            ptagis=_s(r["Tag Code"]), species=_s(r["Species Name"]),
            release_date=_s(r.get("Release Date MMDDYYYY")),
            release_time=_s(r.get("Release Time Time")),
            release_site=_s(r.get("Release Site Name")),
            length_mm=_s(r.get("Mark Length mm")), weight_g=_s(r.get("Mark Weight g")))
    try:
        col = pd.read_csv(PTAGIS_DIR / "collected_v0.csv", dtype=str)
        col["code"] = col["Tag Code"].fillna("").str.upper()
        for code, g in col.groupby("code"):
            g = g.sort_values("Obs Time Value")
            first = g.iloc[0]
            recaptures[code] = dict(
                site=str(first["Site Name"]).strip(),
                time=str(first["Obs Time Value"]).strip(),
                species=str(first["Species Name"]).strip())
    except OSError:
        pass
    return meta, recaptures


def load_track(db, tag, floor_m):
    """Verified item12 chain (identical to fish_figures_2025.py)."""
    con = sqlite3.connect("file:%s?mode=ro" % str(db).replace("\\", "/"), uri=True)
    p = pd.read_sql("select * from tblPositions_Deng where comment = 'solution found'", con)
    con.close()
    p["t0"] = pd.to_numeric(p.ToA, errors="coerce") - pd.to_numeric(p.T01, errors="coerce")
    for c in ("X", "Y", "Z"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    raw = p[(p.solution == "B") & (p.Tag_ID == tag) & p[["X", "Y", "Z", "t0"]].notna().all(axis=1)]
    sub = raw[(raw.Z <= 0) & (raw.Z >= floor_m)]
    g = legacy_receiver_set_consensus(sub)
    gs, _ = smooth_track(g, jump_limit_m=10.0, jump_window=11, smooth_window=3)
    gk, _ = kalman_filter_track(gs, process_noise=2.0, measurement_noise=1.0,
                                z_noise=0.5, max_gap_s=20.0)
    gk = gk.sort_values("t0").reset_index(drop=True)
    dt = gk.t0.diff()
    d3 = np.hypot(gk.x.diff(), gk.y.diff())
    gk["speed_m_s"] = np.where((dt > 0) & (dt < 300), d3 / dt, np.nan)
    return gk, int(len(raw)), int(len(sub))


def temp_at_depth(station, t_epoch, depth_m):
    """Linear-in-time + linear-in-depth temperature (C). depth_m = -Z (0 at surface)."""
    cols = list(station.keys())
    t0 = station[cols[0]][0]
    dt_s = station[cols[0]][1]
    depths = np.array([col_depth_m(c) for c in cols])
    idx = (np.asarray(t_epoch, dtype=float) - t0) / dt_s
    ti_all = [np.interp(idx, np.arange(len(station[c][2])), station[c][2]) for c in cols]
    d = np.clip(np.asarray(depth_m, dtype=float), depths[0], depths[-1])
    seg = np.clip(np.searchsorted(depths, d) - 1, 0, len(depths) - 2)
    w = (d - depths[seg]) / (depths[seg + 1] - depths[seg])
    out = ti_all[0].copy()
    for k in range(len(depths) - 1):
        m = seg == k
        out[m] = ti_all[k][m] * (1 - w[m]) + ti_all[k + 1][m] * w[m]
    return out


def parse_local_datetime(date_str, time_str):
    """'6/17/2025' + '6/17/2025 11:33' -> wallclock epoch (or None).

    The parsed fields are treated as wall clock directly (as-UTC epoch), matching
    the study time basis — NOT via Timestamp.timestamp(), which would apply the
    machine's local timezone (off by 1 h: MDT vs PDT in June)."""
    try:
        dt = pd.to_datetime("%s %s" % (date_str, time_str.split(" ", 1)[-1]))
        ms = pd.DatetimeIndex([dt]).astype("datetime64[ms]").astype(np.int64)[0]
        return float(ms) / 1000.0
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- fate
def compute_fate(gk, cfd_centroid, recap):
    """v1/v2 fate rules + dam PIT-antenna override. Returns (row, details)."""
    first_t, last_t = float(gk.t0.iloc[0]), float(gk.t0.iloc[-1])
    first_x, first_y = float(gk.x.iloc[0]), float(gk.y.iloc[0])
    last_x, last_y = float(gk.x.iloc[-1]), float(gk.y.iloc[-1])
    span_h = (last_t - first_t) / 3600.0

    dists = np.hypot(gk.x - cfd_centroid[0], gk.y - cfd_centroid[1])
    mi = int(dists.to_numpy().argmin())
    closest = dict(x=float(gk.x.iloc[mi]), y=float(gk.y.iloc[mi]),
                   t=float(gk.t0.iloc[mi]), dist=float(dists.iloc[mi]))

    def zone(y):
        if y > FOREBAY_Y:
            return "FOREBAY"
        if y < DAM_FACE_Y:
            return "DAM FACE"
        return "TRANSITION"

    first_in_forebay = first_y > FOREBAY_Y
    last_in_tailrace = last_y < DAM_FACE_Y
    had_crossing = False
    if first_in_forebay:
        below = (gk.y < DAM_FACE_Y).astype(int).to_numpy()
        run = max_run = 0
        for v in below:
            run = run + 1 if v else 0
            max_run = max(max_run, run)
        had_crossing = max_run >= 3

    if first_in_forebay and last_in_tailrace and had_crossing:
        fate = "PASSED"
    elif last_y > FOREBAY_Y:
        fate = "FOREBAY-LOST"
    else:
        fate = "UNKNOWN"
    pit = None
    if recap:
        pit = recap
        fate = "REACHED DAM"  # PIT-antenna detection at the dam (passage not array-confirmed)

    n = min(10, len(gk) - 1)
    if n >= 3:
        seg = gk.iloc[-(n + 1):-1]
        y_std = float(seg.y.std())
        dy = float(seg.y.diff().mean())
        if y_std < 5:
            trend = "STATIONARY"
        elif dy < -1:
            trend = "TOWARD DAM"
        elif dy > 1:
            trend = "AWAY FROM DAM"
        else:
            trend = "PARALLEL"
    else:
        trend = "INSUFFICIENT DATA"

    row = dict(span_h=round(span_h, 1),
               first_t=first_t, last_t=last_t,
               first_xy=(first_x, first_y), last_xy=(last_x, last_y),
               entry_zone=zone(first_y), last_zone=zone(last_y),
               closest=closest, fate=fate, trend=trend, pit=pit,
               n_kalman=int(len(gk)))
    return row


# --------------------------------------------------------------------------- html
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cowlitz Falls early fish 2025 - interactive dashboard</title>
<script>__PLOTLYJS__</script>
<style>
 body { font-family: "Segoe UI", Arial, sans-serif; margin: 14px 22px; color: #222; background: #fafbfc; }
 h1 { font-size: 21px; margin-bottom: 2px; }
 h2 { font-size: 15px; margin: 18px 0 6px 0; color: #2c3e50; border-bottom: 1px solid #d5dbe0; padding-bottom: 3px; }
 .sub { color: #667; font-size: 12.5px; margin-bottom: 10px; }
 #roster { border-collapse: collapse; width: 100%; font-size: 12.5px; background: #fff; }
 #roster th { background: #34495e; color: #fff; padding: 5px 8px; text-align: left; font-size: 12px; }
 #roster td { padding: 4px 8px; border-bottom: 1px solid #e3e8ec; }
 #roster tr.sel { background: #eaf2fb; }
 #roster tr { cursor: pointer; }
 #roster tr:hover { background: #f2f7fd; }
 .fate { font-weight: 600; }
 #filters { background: #fff; border: 1px solid #d5dbe0; border-radius: 6px; padding: 8px 12px;
            font-size: 12.5px; display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; }
 #filters label { color: #445; }
 select, input[type=number], input[type=datetime-local] { font-size: 12.5px; padding: 2px 4px; }
 #nshown { font-weight: 600; color: #1a6b3c; }
 button { font-size: 12.5px; padding: 3px 10px; cursor: pointer; }
 .sliderrow { margin: 4px 2px 0 2px; font-size: 12px; color: #445; }
 .sliderrow input[type=range] { width: 60vw; }
 #notes { font-size: 12.5px; color: #345; background: #fff; border: 1px solid #d5dbe0;
          border-radius: 6px; padding: 8px 14px; line-height: 1.45; }
 #notes b { color: #223; }
 .panelfull { width: 100%; }
 .half { width: 49.5%; display: inline-block; vertical-align: top; }
</style>
</head>
<body>
<h1>Cowlitz Falls Dam - early fish acoustic tracking 2025 (interactive)</h1>
<div class="sub">4 tagged study fish (3 confirmed juvenile Chinook; 26E4 not in the release file) released upstream
and tracked in the forebay array. All times local (PDT, UTC-7) - the study time basis; UTC = local + 7 h (shown on hover).
Click a fish row to focus it.</div>

<h2>1 &middot; Fish summary</h2>
<table id="roster"><thead><tr>
<th>Tag</th><th>Species</th><th>PTAGIS code</th><th>Release (local)</th><th>Fork length</th><th>Weight</th>
<th>First detected</th><th>Last detected</th><th>Hours in array</th><th>Fate</th><th>Dam PIT-antenna detection</th>
</tr></thead><tbody>
__ROSTER_ROWS__
</tbody></table>

<h2>2 &middot; Filters (combine any of them)</h2>
<div id="filters">
 <label>Species
  <select id="fSpecies"><option>All</option><option>Chinook</option><option>Unconfirmed</option></select></label>
 <label>Tag
  <select id="fTag"></select></label>
 <label>From <input type="datetime-local" id="fT0" step="300"></label>
 <label>To <input type="datetime-local" id="fT1" step="300"></label>
 <label><input type="checkbox" id="fTempOn"> Water temp between
  <input type="number" id="fTminC" step="0.5" style="width:52px"> and
  <input type="number" id="fTmaxC" step="0.5" style="width:52px"> &deg;C at
  <select id="fTempRef">
    <option value="fish">fish depth</option>
    <option value="0.5">0.5 m (surface)</option>
    <option value="1.5">1.5 m</option>
    <option value="9">9 m</option>
    <option value="18">18 m</option>
  </select></label>
 <label>Temp station (panel 4)
  <select id="fStation"><option>DD_N</option><option>FBS</option><option>PDS</option><option>DD_S</option></select></label>
 <label><input type="checkbox" id="fPts"> show points</label>
 <button id="fReset">reset</button>
 <span id="nshown"></span>
</div>
<div class="sliderrow">Time window:
 <input type="range" id="sT0" min="0" max="1000" value="0">
 <input type="range" id="sT1" min="0" max="1000" value="1000">
 <span id="slabels"></span></div>

<h2>3 &middot; 3D movement</h2>
<div id="fig3d" style="height:620px"></div>
<div class="sub">Line = Kalman-filtered track (green &gt; first, red &lt; last). Grey x = receivers.
Blue plane = water surface, tan plane = bay-bottom floor gate. Hover any point for time, depth, speed, water temperature.</div>

<h2>4 &middot; Water column &amp; fish depth</h2>
<div class="half"><div id="figtemp" style="height:400px"></div>
<div class="sub">Water temperature vs time (selected station, one line per sensor depth).
Shaded bands = each fish's observation window; dashed lines = release times.</div></div>
<div class="half"><div id="figdepth" style="height:400px"></div>
<div class="sub">Fish depth vs time (surface at top). With one tag selected, the points are coloured by
water temperature at the fish's depth (viridis, see bar).</div></div>

<h2>5 &middot; Notes</h2>
<div id="notes">
<b>How to read this.</b> A point is drawn only if the fish passes the species/tag filter, its time is inside
the From-To window, and (when the water-temp checkbox is on) the temperature at the chosen depth is inside
the &deg;C range. Use the tag selector to view one fish at a time, or "All tags" to compare.
<br><br>
<b>Fates.</b> <span class="fate" style="color:#16a085">REACHED DAM</span> = PIT-antenna detection at the dam site
(passage through the dam not confirmed by the acoustic array, which sits entirely upstream of the dam).
<span class="fate" style="color:#e74c3c">FOREBAY-LOST</span> = last seen in the forebay, no dam detection.
<span class="fate" style="color:#f39c12">UNKNOWN</span> = last seen in the transition band.
"DAM FACE (Y&lt;75)" is a proxy for the array edge nearest the dam; the whole array is upstream of the dam
(CFD receivers are in the forebay per the config workbook), so a fish passing the dam leaves the array footprint.
<br><br>
<b>Position quality.</b> X/Y are the validated dimension (GPS-truthed drag tests: median 2.4-10.7 m). Z is less
reliable (systematic bias ~1.6-3.7 m deeper than truth; flat-array mirror artifacts removed by the [floor, 0] m
gate, floor = __FLOOR__ m from the run file). Receivers ZOI03/ZOI06 had no raw data until ~06-27, so early
windows (1D5A) have sparser coverage. CFD01/ZOI11 clocks run ~940 ms late (uncorrected).
<br><br>
<b>Sources.</b> Positions: jsats3d_2025_earlyfish.db (root-B Deng 3D, consensus+smooth+Kalman chain, verified 2026-10-25).
Species/release/recapture: PTAGIS released_v0 / collected_v0. Water temperature: 2025_Temp_String_Data_5min_interpolated.csv
(5-min HOBO, stations DD_N/FBS/PDS/DD_S; per-point local temp from DD_N). WSEL: tblWSEL. 26E4 is not in the PTAGIS
release file (species unconfirmed; cohort context suggests Chinook).
</div>

<script>
window.__errs = [];
window.addEventListener("error", e => {
  window.__errs.push("error: " + e.message + " @ " + (e.filename||"") + ":" + e.lineno);
});
window.addEventListener("unhandledrejection", e => {
  window.__errs.push("rejection: " + ((e.reason && e.reason.message) || e.reason));
});
const D = __DATA__;
const PLOT = {template: {paper_bgcolor: "#fafbfc", plot_bgcolor: "#ffffff",
  font: {size: 11, family: "Segoe UI, Arial"}}};

function f32(b64){ const s = atob(b64); const u = new Uint8Array(s.length);
  for (let i=0;i<s.length;i++) u[i] = s.charCodeAt(i); return new Float32Array(u.buffer); }
function f64(b64){ const s = atob(b64); const u = new Uint8Array(s.length);
  for (let i=0;i<s.length;i++) u[i] = s.charCodeAt(i); return new Float64Array(u.buffer); }
const pad = n => (n<10?"0":"") + n;
function localStr(e){ const d = new Date(e*1000);
  return pad(d.getUTCMonth()+1)+"-"+pad(d.getUTCDate())+" "+pad(d.getUTCHours())+":"+pad(d.getUTCMinutes()); }
const utcStr = e => localStr(e + 25200);

// decode track arrays
for (const f of Object.values(D.fish)) {
  f.t = f64(f.b.t); f.x = f32(f.b.x); f.y = f32(f.b.y); f.z = f32(f.b.z); f.speed = f32(f.b.speed);
  f.tempd = {"0.5": f32(f.b.t05), "1.5": f32(f.b.t15), "9": f32(f.b.t9), "18": f32(f.b.t18)};
}
// decode covariate series
for (const st of Object.values(D.temp)) { st.cols = {}; for (const k in st.cols_b) st.cols[k] = f32(st.cols_b[k]); }
D.wsel.v = f32(D.wsel.b64);
// Plotly 3.x date axes always render in the VIEWER's local timezone (the axis-level
// tz attribute was dropped). Stored epochs are PDT wall clock encoded as-UTC (study
// basis), so compensate with the viewer's offset: every axis then reads the study
// wall clock no matter where the file is opened (range is Jun-Sep, DST-stable).
const TZOFF_MS = new Date(1750000000000).getTimezoneOffset() * 60000;
const ax = e => e * 1000 + TZOFF_MS;   // epoch seconds (wall clock) -> axis ms
const TDD = [0.5, 1.5, 9, 18];
function tempAtDepth(f, i, ref){
  if (ref === "fish") {
    const d = Math.min(18, Math.max(0.5, -f.z[i]));
    let k = 0; while (k < 3 && d > TDD[k+1]) k++;
    const a = f.tempd[String(TDD[k])][i], b = f.tempd[String(TDD[k+1])][i];
    if (isNaN(a) || isNaN(b)) return NaN;
    const w = (d - TDD[k]) / (TDD[k+1] - TDD[k]);
    return a*(1-w) + b*w;
  }
  return f.tempd[ref][i];
}

const state = {species: "All", tag: "1D5A", t0: D.tmin, t1: D.tmax,
               tempOn: false, tminC: D.tminC, tmaxC: D.tmaxC, tempRef: "fish",
               station: "DD_N", showPts: false};
const FATE_C = {"PASSED": "#27ae60", "REACHED DAM": "#16a085", "FOREBAY-LOST": "#e74c3c", "UNKNOWN": "#f39c12"};

function fishVisible(tag){
  const f = D.fish[tag];
  if (!f) return false;
  if (state.species !== "All" && f.spkey !== state.species) return false;
  if (state.tag !== "All" && tag !== state.tag) return false;
  return true;
}
function maskOf(f){
  const m = new Uint8Array(f.t.length); let n = 0;
  for (let i=0;i<f.t.length;i++){
    if (f.t[i] < state.t0 || f.t[i] > state.t1) continue;
    if (state.tempOn) { const tv = tempAtDepth(f, i, state.tempRef);
      if (isNaN(tv) || tv < state.tminC || tv > state.tmaxC) continue; }
    m[i] = 1; n++;
  }
  return [m, n];
}

function render3d(){
  const tr = [];
  // water surface + floor planes
  const nx = 24, ny = 24, xs = [], ys = [];
  for (let i=0;i<=nx;i++) xs.push(D.sx0 + i*(D.sx1-D.sx0)/nx);
  for (let j=0;j<=ny;j++) ys.push(D.sy0 + j*(D.sy1-D.sy0)/ny);
  const X = ys.map(() => xs.slice());
  const Y = ys.map(y => new Array(xs.length).fill(y));
  const zs0 = ys.map(() => new Array(xs.length).fill(0));
  const zsf = ys.map(() => new Array(xs.length).fill(D.floor));
  // NOTE: surfacecolor must be a plain color in Plotly 3.x — a gradient array
  // ([[0,"#..."]]) throws inside Plotly's surface colorization.
  tr.push({type:"surface", x:X, y:Y, z:zs0, opacity:0.16, showscales:false, hoverinfo:"skip",
           surfacecolor:"#7fb3d5", name:"water surface", showlegend:false});
  tr.push({type:"surface", x:X, y:Y, z:zsf, opacity:0.14, showscales:false, hoverinfo:"skip",
           surfacecolor:"#d8c9a3", name:"floor gate", showlegend:false});
  // receivers
  tr.push({type:"scatter3d", mode:"markers+text",
    x:Object.values(D.rec).map(r=>r[0]), y:Object.values(D.rec).map(r=>r[1]), z:Object.values(D.rec).map(r=>r[2]),
    text:Object.keys(D.rec), textfont:{size:8,color:"#8a8f98"}, textposition:"top center",
    marker:{symbol:"x",size:5,color:"#8a8f98"}, hoverinfo:"skip", showlegend:false});
  // fish
  for (const tag of Object.keys(D.fish)) {
    if (!fishVisible(tag)) continue;
    const f = D.fish[tag]; const [m, n] = maskOf(f);
    if (!n) continue;
    const x=[],y=[],z=[],tt=[],hd=[];
    for (let i=0;i<f.t.length;i++) if (m[i]) {
      x.push(f.x[i]); y.push(f.y[i]); z.push(f.z[i]); tt.push(f.t[i]);
      const tv = tempAtDepth(f, i, "fish");
      hd.push([localStr(f.t[i]), utcStr(f.t[i]),
               (isNaN(tv)?"-":tv.toFixed(1)+" C"), (isNaN(f.speed[i])?"-":f.speed[i].toFixed(2)+" m/s")]);
    }
    tr.push({type:"scatter3d", mode:"lines", x,y,z,
      line:{color:f.color, width:5}, name:f.label, hoverinfo:"skip"});
    if (state.showPts)
      tr.push({type:"scatter3d", mode:"markers", x,y,z, customdata:hd,
        marker:{size:3, opacity:0.9, color:tt, cmin:f.first, cmax:f.last, colorscale:"Viridis", showscale:false},
        hovertemplate:"%{customdata[0]} local<br>%{customdata[1]} UTC<br>temp %{customdata[2]}  speed %{customdata[3]}<extra>"+f.tag+"</extra>",
        showlegend:false});
    let fi=-1, li=-1;
    for (let i=0;i<f.t.length;i++){ if (m[i]){ if (fi<0) fi=i; li=i; } }
    tr.push({type:"scatter3d", mode:"markers", x:[f.x[fi]], y:[f.y[fi]], z:[f.z[fi]],
      marker:{symbol:"triangle-right", size:8, color:"#2ecc71"}, hoverinfo:"skip", showlegend:false});
    tr.push({type:"scatter3d", mode:"markers", x:[f.x[li]], y:[f.y[li]], z:[f.z[li]],
      marker:{symbol:"triangle-left", size:8, color:"#e74c3c"}, hoverinfo:"skip", showlegend:false});
    const off = D.labelOff[tag];
    const lx = f.x[li]+off[0], ly = f.y[li]+off[1], lz = f.z[li]+off[2];
    tr.push({type:"scatter3d", mode:"lines", x:[f.x[li],lx], y:[f.y[li],ly], z:[f.z[li],lz],
      line:{width:2,color:f.color,opacity:0.45}, hoverinfo:"skip", showlegend:false});
    tr.push({type:"scatter3d", mode:"text", x:[lx], y:[ly], z:[lz], text:[f.label],
      textfont:{size:13,color:f.color,family:"Arial, sans-serif"}, hoverinfo:"skip", showlegend:false});
  }
  Plotly.react("fig3d", tr, Object.assign({
    margin:{l:0,r:0,t:28,b:0},
    scene:{ xaxis:{title:"X (m)", range:[D.sx0,D.sx1]}, yaxis:{title:"Y (m)", range:[D.sy0,D.sy1]},
            zaxis:{title:"Z (m)", range:[D.floor, 2]},
            camera:{eye:{x:1.6,y:-1.6,z:0.7}} },
    showlegend:true, legend:{orientation:"h", x:0, y:1.06},
  }, PLOT.template), {responsive:true, displaylogo:false});
}

function tempLayout(){
  const st = D.temp[state.station];
  const tmin = st.t0, dt = st.dt;
  const tr = [];
  const pal = ["#6baed6","#3182bd","#08519c","#08306b","#05133d","#03081a"];
  let k = 0;
  for (const c in st.cols) {
    const v = st.cols[c];
    const x = []; for (let i=0;i<v.length;i++) x.push(ax(tmin + i*dt));
    tr.push({type:"scatter", mode:"lines", x, y:v,
      line:{width:1.6,color:pal[Math.min(k, pal.length-1)]}, name:c, hoverinfo:"skip"});
    k++;
  }
  // WSEL on secondary axis
  const wx = [], wv = D.wsel.v;
  for (let i=0;i<wv.length;i++) wx.push(ax(D.wsel.t0 + i*D.wsel.dt));
  tr.push({type:"scatter", mode:"lines", x:wx, y:wv, yaxis:"y2",
    line:{width:1.2,color:"#95a5a6",dash:"dot"}, name:"WSEL (ft)", hoverinfo:"skip"});
  const shapes = [], anns = [];
  for (const tag of Object.keys(D.fish)) {
    if (!fishVisible(tag)) continue;
    const f = D.fish[tag];
    shapes.push({type:"rect", x0:ax(f.first), x1:ax(f.last), y0:0, y1:1,
      yref:"paper", fillcolor:f.color, opacity:0.10, line:{width:0}});
    anns.push({x:ax(f.first), y:1.02, yref:"paper", text:f.tag, showarrow:false,
      font:{size:11,color:f.color,weight:"bold"}});
    if (f.releaseT) shapes.push({type:"line", x0:ax(f.releaseT), x1:ax(f.releaseT),
      y0:0, y1:1, yref:"paper", line:{dash:"dash", width:1.5, color:f.color}});
  }
  return {tr, shapes, anns};
}
function renderTemp(){
  const {tr, shapes, anns} = tempLayout();
  Plotly.react("figtemp", tr, Object.assign({
    margin:{l:42,r:10,t:26,b:34},
    // x values are passed through ax() so the axis reads study wall clock (PDT)
    // regardless of the viewer's timezone (Plotly 3.x has no axis tz attribute).
    xaxis:{type:"date", tickformat:"%m-%d %H:%M", title:"Time (local, PDT)"},
    yaxis:{title:"Temp (C)"},
    yaxis2:{title:"WSEL (ft)", overlaying:"y", side:"right", showgrid:false},
    shapes, annotations:anns,
    legend:{orientation:"h", x:0, y:-0.18},
    title:{text:"Water column - station " + state.station, font:{size:12}},
  }, PLOT.template), {responsive:true, displaylogo:false});
}
function renderDepth(){
  const tr = [];
  for (const tag of Object.keys(D.fish)) {
    if (!fishVisible(tag)) continue;
    const f = D.fish[tag]; const [m, n] = maskOf(f);
    if (!n) continue;
    const x=[], y=[], tv=[];
    for (let i=0;i<f.t.length;i++) if (m[i]) { x.push(ax(f.t[i])); y.push(f.z[i]); tv.push(tempAtDepth(f,i,"fish")); }
    if (state.tag === tag) {
      tr.push({type:"scatter", mode:"lines", x, y,
        line:{width:2.5, color:f.color, opacity:0.5}, showlegend:false, hoverinfo:"skip"});
      tr.push({type:"scatter", mode:"markers", x, y,
        marker:{size:3.5, color:tv, colorscale:"Viridis", cmin:D.tminC, cmax:D.tmaxC, showscale:true,
                colorbar:{title:{text:"C", side:"right"}, thickness:12, len:0.8}},
        name:f.label + " - water temp at fish depth", hoverinfo:"skip"});
    } else {
      tr.push({type:"scatter", mode:"lines", x, y, line:{width:2.4,color:f.color},
        name:f.label, hoverinfo:"skip"});
    }
  }
  Plotly.react("figdepth", tr, Object.assign({
    margin:{l:42,r:10,t:26,b:34},
    xaxis:{type:"date", tickformat:"%m-%d %H:%M", title:"Time (local, PDT)"},
    yaxis:{title:"Depth (m)", autorange:"reversed", zeroline:false},
    legend:{orientation:"h", x:0, y:-0.18},
    title:{text:"Fish depth over time", font:{size:12}},
  }, PLOT.template), {responsive:true, displaylogo:false});
}
function updateReadout(){
  let n = 0, nf = 0;
  for (const tag of Object.keys(D.fish)) if (fishVisible(tag)) { nf++; n += maskOf(D.fish[tag])[1]; }
  document.getElementById("nshown").textContent = "showing " + n + " track points for " + nf + " fish";
  document.querySelectorAll("#roster tbody tr").forEach(tr =>
    tr.classList.toggle("sel", tr.dataset.tag === state.tag));
}
function syncSliders(){
  const s0 = document.getElementById("sT0"), s1 = document.getElementById("sT1");
  s0.value = Math.round(1000*(state.t0-D.tmin)/(D.tmax-D.tmin));
  s1.value = Math.round(1000*(state.t1-D.tmin)/(D.tmax-D.tmin));
  document.getElementById("slabels").textContent = localStr(state.t0) + "  ->  " + localStr(state.t1);
}
function renderAll(){
  // Per-step isolation: a failure in one panel (typically the page's very first Plotly
  // figure, a font/canvas init race in Plotly's text rasterization) must not block the
  // others. Returns the number of failed steps.
  const steps = [["render3d", render3d], ["renderTemp", renderTemp], ["renderDepth", renderDepth],
                 ["updateReadout", updateReadout], ["syncSliders", syncSliders]];
  let fails = 0;
  for (const s of steps) {
    try { s[1](); }
    catch (e) { fails++; window.__errs.push(s[0] + ": " + e.message); }
  }
  return fails;
}

function bind(){
  const $ = id => document.getElementById(id);
  const tagSel = $("fTag");
  tagSel.innerHTML = '<option>All</option>' +
    Object.keys(D.fish).map(t => '<option value="'+t+'">'+D.fish[t].label+'</option>').join("");
  tagSel.value = state.tag;
  const toInput = e => pad(new Date(e).getUTCFullYear())+"-"+pad(new Date(e).getUTCMonth()+1)+"-"+
    pad(new Date(e).getUTCDate())+"T"+pad(new Date(e).getUTCHours())+":"+pad(new Date(e).getUTCMinutes());
  const i0 = $("fT0"), i1 = $("fT1");
  i0.min = i1.min = toInput(D.tmin*1000); i0.max = i1.max = toInput(D.tmax*1000);
  i0.value = toInput(state.t0*1000); i1.value = toInput(state.t1*1000);
  const fromInput = v => v ? Date.parse(v + "Z")/1000 : null;  // wall clock -> study epoch
  i0.oninput = () => { const v = fromInput(i0.value); if (v && v <= state.t1){ state.t0 = v; renderAll(); } };
  i1.oninput = () => { const v = fromInput(i1.value); if (v && v >= state.t0){ state.t1 = v; renderAll(); } };
  $("sT0").oninput = e => { state.t0 = D.tmin + (e.target.value/1000)*(D.tmax-D.tmin);
    i0.value = toInput(state.t0*1000); renderAll(); };
  $("sT1").oninput = e => { state.t1 = D.tmin + (e.target.value/1000)*(D.tmax-D.tmin);
    i1.value = toInput(state.t1*1000); renderAll(); };
  $("fSpecies").oninput = e => { state.species = e.target.value; renderAll(); };
  tagSel.oninput = e => { state.tag = e.target.value; renderAll(); };
  $("fTempOn").onchange = e => { state.tempOn = e.target.checked; renderAll(); };
  $("fTminC").oninput = e => { state.tminC = parseFloat(e.target.value)||0; renderAll(); };
  $("fTmaxC").oninput = e => { state.tmaxC = parseFloat(e.target.value)||0; renderAll(); };
  $("fTempRef").oninput = e => { state.tempRef = e.target.value; renderAll(); };
  $("fStation").oninput = e => { state.station = e.target.value; renderTemp(); };
  $("fPts").onchange = e => { state.showPts = e.target.checked; render3d(); };
  $("fReset").onclick = () => {
    Object.assign(state, {species:"All", tag:"1D5A", t0:D.tmin, t1:D.tmax, tempOn:false,
      tminC:D.tminC, tmaxC:D.tmaxC, tempRef:"fish", station:"DD_N", showPts:false});
    $("fSpecies").value="All"; tagSel.value="1D5A";
    i0.value=toInput(D.tmin*1000); i1.value=toInput(D.tmax*1000);
    $("fTempOn").checked=false; $("fTminC").value=D.tminC.toFixed(1); $("fTmaxC").value=D.tmaxC.toFixed(1);
    $("fTempRef").value="fish"; $("fStation").value="DD_N"; $("fPts").checked=false;
    renderAll();
  };
  document.querySelectorAll("#roster tbody tr").forEach(tr =>
    tr.onclick = () => { state.tag = tr.dataset.tag; tagSel.value = state.tag; renderAll(); });
  $("fTminC").value = D.tminC.toFixed(1); $("fTmaxC").value = D.tmaxC.toFixed(1);
  // First render must wait until layout gives the figure divs real sizes; rendering a
  // 2D plot before first layout breaks Plotly's internal canvas computations. In some
  // browser environments the very first 2D render also hits a transient Plotly-internal
  // error, so retry on a timer (rAF does not fire in headless/hidden tabs) until it
  // succeeds.
  let tries = 0;
  const firstRender = () => {
    const el = document.getElementById("fig3d");
    if (el && el.clientWidth > 50) {
      const fails = renderAll();
      if (fails === 0) return;            // clean render
      if (++tries < 50) { setTimeout(firstRender, 200); return; }
    }
    renderAll();                          // give up: render anyway
  };
  // Wait for layout AND web fonts before the first render: the page's first Plotly
  // figure can hit a font/canvas init race in Plotly's text rasterization. rAF is not
  // reliable in headless/hidden tabs, so everything here is timer-based.
  const startFirst = () => setTimeout(firstRender, 50);
  const arm = () => {
    if (document.readyState !== "complete") { window.addEventListener("load", startFirst); return; }
    if (document.fonts && document.fonts.ready) {
      const t = setTimeout(startFirst, 1500);   // never block first paint on fonts
      document.fonts.ready.then(() => { clearTimeout(t); startFirst(); });
    } else startFirst();
  };
  arm();
  let rzT = null;
  window.addEventListener("resize", () => {
    if (rzT) clearTimeout(rzT);
    rzT = setTimeout(() => { if (document.getElementById("fig3d").data) renderAll(); }, 250);
  });
}
bind();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    db, out = Path(args.db), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Fish Dashboard v3 (filterable)  db=%s" % db.name)
    print("=" * 72)

    run_file = run_file_from_status(db)
    floor_m, floor_src = resolve_depth_floor(db, None, run_file)
    print("depth floor: %.2f m (%s)" % (floor_m, floor_src))

    # ---- receivers + WSEL
    con = sqlite3.connect("file:%s?mode=ro" % str(db).replace("\\", "/"), uri=True)
    rec = pd.read_sql("select Rec_ID, X_t, Y_t, Z_t from tblReceiver", con).set_index("Rec_ID")
    wsel = pd.read_sql("select timeStamp, WSEL from tblWSEL order by timeStamp", con)
    con.close()
    wsel_t0 = wallclock_epoch(wsel.timeStamp)
    wsel_vals = wsel.WSEL.to_numpy(dtype=np.float32)
    print("WSEL: %d rows  %s -> %s" % (len(wsel), wsel.timeStamp.iloc[0], wsel.timeStamp.iloc[-1]))
    wsel_disp = dict(t0=float(wsel_t0[0]), dt=900.0,
                     b64=b64f32(wsel_vals[::TEMP_DISPLAY_DOWNSAMPLE]))

    # ---- water column temperatures
    print("loading water-column temperatures: %s" % TEMP_CSV.name)
    temp_stations = load_temp_watercolumn()
    for st, cols in temp_stations.items():
        c0 = cols[list(cols)[0]]
        print("  %-4s %s  %d depths  t0=%.0f" % (st, list(cols.keys()), len(cols), c0[0]))

    # ---- tag metadata (PTAGIS)
    meta, recaptures = load_ptagis_metadata()
    print("PTAGIS: %d acoustic tags, %d recaptured 3DD codes" % (len(meta), len(recaptures)))

    # ---- tracks (verified chain, self-checked)
    cfd = rec.loc[[r for r in rec.index if r.startswith("CFD")]]
    cfd_centroid = (float(cfd.X_t.mean()), float(cfd.Y_t.mean()))
    tracks, fate_rows = {}, []
    for tag in STUDY_TAGS:
        gk, n_raw, n_gate = load_track(db, tag, floor_m)
        if gk.empty:
            print("  %s: NO TRACK (n_raw=%d)" % (tag, n_raw))
            continue
        n = len(gk)
        ok = EXPECTED_KALMAN.get(tag)
        print("  %s: raw=%d gated=%d kalman=%d  %s"
              % (tag, n_raw, n_gate, n,
                 "OK (matches verified fish_qc)" if ok == n else "MISMATCH vs %s" % ok))
        if ok != n:
            raise SystemExit("Kalman count mismatch for %s - chain drift, aborting." % tag)
        m = meta.get(tag.upper())
        recap = None
        if m and m["ptagis"]:
            recap = recaptures.get(m["ptagis"].upper())
        fate = compute_fate(gk, cfd_centroid, recap)
        fate["tag"] = tag
        fate["meta"] = m
        tracks[tag] = gk
        fate_rows.append(fate)
        pit = ("  PIT: %s @ %s (%s)" % (fate["pit"]["time"], fate["pit"]["site"], tag)
               if recap else "")
        print("     fate=%s  trend=%s  span=%.1f h  closest=%.1f m%s"
              % (fate["fate"], fate["trend"], fate["span_h"], fate["closest"]["dist"], pit))

    # ---- per-point temperatures + JS data
    dd = temp_stations[TEMP_REF_DEPTH]
    all_temps = []
    fish_js = {}
    roster_rows, track_rows = [], []
    for fr in fate_rows:
        tag = fr["tag"]
        gk = tracks[tag]
        t_e = gk.t0.to_numpy(dtype=float)
        depth_m = -gk.z.to_numpy(dtype=float)
        temp_fish = temp_at_depth(dd, t_e, depth_m)
        temp_fixed = {c: temp_at_depth(dd, t_e, np.full(len(t_e), col_depth_m(c)))
                      for c in TEMP_STATIONS[TEMP_REF_DEPTH]}
        all_temps.append(temp_fish)
        m = fr["meta"]
        species = m["species"] if m else "Unconfirmed (not in PTAGIS)"
        spkey = "Chinook" if (m and m["species"].lower() == "chinook") else "Unconfirmed"
        rel_t = None
        if m:
            rel_t = parse_local_datetime(m["release_date"], m["release_time"])
        release_txt = ("%s %s" % (m["release_date"], m["release_time"].split(" ")[-1])) if m else "not in file"
        length_txt = ("%s mm" % m["length_mm"]) if (m and m["length_mm"]) else "-"
        weight_txt = ("%s g" % m["weight_g"]) if (m and m["weight_g"]) else "-"
        pit_txt = ("%s @ %s" % (fr["pit"]["time"], fr["pit"]["site"])) if fr["pit"] else "-"
        label = "%s - %s" % (tag, (m["species"] if m else "unconfirmed"))
        fish_js[tag] = dict(
            tag=tag, label=label, spkey=spkey, species=species,
            ptagis=m["ptagis"] if m else "-",
            release=release_txt, releaseT=rel_t,
            length_mm=m["length_mm"] if m else "-", weight_g=m["weight_g"] if m else "-",
            fate=fr["fate"], first=float(gk.t0.iloc[0]), last=float(gk.t0.iloc[-1]),
            color=TAG_COLORS[tag],
            b=dict(t=b64f64(t_e), x=b64f32(gk.x.to_numpy(dtype=np.float32)),
                   y=b64f32(gk.y.to_numpy(dtype=np.float32)),
                   z=b64f32(gk.z.to_numpy(dtype=np.float32)),
                   speed=b64f32(gk.speed_m_s.fillna(0).to_numpy(dtype=np.float32)),
                   t05=b64f32(temp_fixed["DD_N_0p5"]), t15=b64f32(temp_fixed["DD_N_1p5"]),
                   t9=b64f32(temp_fixed["DD_N_9"]), t18=b64f32(temp_fixed["DD_N_18"])))
        roster_rows.append(
            "<tr data-tag=\"%s\"><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td>"
            "<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%.1f</td>"
            "<td class=\"fate\" style=\"color:%s\">%s</td><td>%s</td></tr>"
            % (tag, tag, species, fish_js[tag]["ptagis"], release_txt,
               length_txt, weight_txt,
               pd.to_datetime(t_e[0], unit="s").strftime("%m-%d %H:%M"),
               pd.to_datetime(t_e[-1], unit="s").strftime("%m-%d %H:%M"),
               fr["span_h"], FATE_COLORS[fr["fate"]], fr["fate"], pit_txt))
        sub = pd.DataFrame({"tag": tag, "t_epoch": t_e,
                            "t_local": pd.to_datetime(t_e, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                            "t_utc": pd.to_datetime(t_e + 7 * 3600, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                            "x_m": gk.x.round(2), "y_m": gk.y.round(2), "z_m": gk.z.round(2),
                            "speed_m_s": gk.speed_m_s.round(3),
                            "temp_at_fish_depth_C": np.round(temp_fish, 2),
                            "temp_surface_0p5m_C": np.round(temp_fixed["DD_N_0p5"], 2)})
        track_rows.append(sub)
        print("  %s: temp range %.1f-%.1f C over track" % (tag, np.nanmin(temp_fish), np.nanmax(temp_fish)))

    all_temps = np.concatenate([t for t in all_temps if len(t)])
    tminC, tmaxC = float(np.nanmin(all_temps)), float(np.nanmax(all_temps))

    # ---- embed covariates (downsampled display series)
    temp_js = {}
    for st, cols in temp_stations.items():
        c0 = cols[list(cols)[0]]
        temp_js[st] = dict(t0=c0[0], dt=c0[1] * TEMP_DISPLAY_DOWNSAMPLE,
                           cols_b={c: b64f32(v[2][::TEMP_DISPLAY_DOWNSAMPLE]) for c, v in cols.items()})
    tmin = min(f["first"] for f in fish_js.values())
    tmax = max(f["last"] for f in fish_js.values())
    data = dict(
        fish=fish_js,
        rec={r: [float(a), float(b), float(c)] for r, (a, b, c) in
             [(i, (rec.X_t[i], rec.Y_t[i], rec.Z_t[i])) for i in rec.index]},
        temp=temp_js, wsel=wsel_disp,
        tmin=float(tmin), tmax=float(tmax), tminC=round(tminC, 1), tmaxC=round(tmaxC, 1),
        floor=float(floor_m),
        sx0=float(rec.X_t.min() - 25), sx1=float(rec.X_t.max() + 25),
        sy0=float(rec.Y_t.min() - 25), sy1=float(rec.Y_t.max() + 25),
        labelOff={"26E4": [-10, 6, 1.5], "9F8C": [-10, -6, 1.5],
                  "1D5A": [9, -7, 1.5], "B5ED": [9, 7, 1.5]})

    # ---- write HTML
    import plotly.offline as poff
    html = (HTML_TEMPLATE
            .replace("__PLOTLYJS__", poff.get_plotlyjs())
            .replace("__DATA__", json.dumps(data))
            .replace("__ROSTER_ROWS__", "\n".join(roster_rows))
            .replace("__FLOOR__", "%.1f" % floor_m))
    (out / "dashboard.html").write_text(html, encoding="utf-8")
    print("dashboard.html: %.1f MB" % ((out / "dashboard.html").stat().st_size / 1e6))

    # ---- CSVs + README
    tracks_df = pd.concat(track_rows, ignore_index=True)
    tracks_df.to_csv(out / "fish_tracks.csv", index=False)
    roster = pd.DataFrame([dict(
        tag=f["tag"], species=f["species"], ptagis=f["ptagis"], release=f["release"],
        length_mm=f["length_mm"], weight_g=f["weight_g"], fate=f["fate"],
        first_det_local=pd.to_datetime(f["first"], unit="s").strftime("%Y-%m-%d %H:%M"),
        last_det_local=pd.to_datetime(f["last"], unit="s").strftime("%Y-%m-%d %H:%M"),
        hours_in_array=round((f["last"] - f["first"]) / 3600.0, 1))
        for f in fish_js.values()])
    roster.to_csv(out / "fish_roster.csv", index=False)

    readme = """# Fish Dashboard v3 - filterable interactive view (earlyfish 2025)

Built by `scripts/fish_dashboard_v3.py` (re-run: `python scripts/fish_dashboard_v3.py`).

## Files
- `dashboard.html` - the dashboard (open in any browser; self-contained, no server)
- `fish_roster.csv` - per-fish metadata (species, release, fate)
- `fish_tracks.csv` - per-point tracks with local water temperature

## What it does
Combines the 4 tagged-fish Kalman tracks with water-column temperature (5-min HOBO),
WSEL, receiver layout, and PTAGIS metadata. Filters (species, tag, time window,
water temperature at a chosen depth) combine; the 3D view shows one fish by default
or all of them, with clean endpoint labels.

## Time basis
Local (PDT, UTC-7) - the study time basis (position epochs, temperature and WSEL
timestamps are all PDT wall clock; journals 2026-09-24/25, GPS-anchored 06-10 drag
test). UTC = local + 7 h (hover shows both). Note: the v2 dashboard shows the same
moments in UTC; times differ by exactly 7 h between the two.

## Track chain
Identical to the verified fish_figures_2025.py chain (root-B -> plausibility gate
[floor, 0] -> consensus -> smooth -> Kalman); Kalman counts are self-checked against
the verified fish_qc.csv at build time (abort on mismatch).

## Caveats
- X/Y validated (drag medians 2.4-10.7 m); Z biased ~1.6-3.7 m deep + artifacts gated.
- ZOI03/ZOI06 raw data missing until ~06-27 (early windows sparse).
- CFD01/ZOI11 clocks ~940 ms late (uncorrected).
- 26E4 not in the PTAGIS release file (species unconfirmed).
- "DAM FACE (Y<75)" is a proxy; the whole array is upstream of the dam.
- 1D5A was PIT-detected at the dam site 06-19 10:06 (PTAGIS collected_v0), 12 min
  before its last acoustic detection - treated as REACHED DAM (passage not
  array-confirmed).
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print("wrote: dashboard.html, fish_tracks.csv (%d rows), fish_roster.csv, README.md"
          % len(tracks_df))


if __name__ == "__main__":
    main()
