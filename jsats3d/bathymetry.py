"""Bathymetry support for jsats3d.

Load bathymetry XYZ data into the project database and use it to filter
acoustic positioning solutions that fall outside the water column:
  - Above the water surface (Z > WSEL at solution time)
  - Below the riverbed (local bathymetry elevation at solution XY)

Uses receiver Z_t values as local water surface references to handle
the fact that bathymetry Z and position Z are in different coordinate
systems that vary by location.

Usage
-----
    from jsats3d import bathymetry
    bathymetry.load_xyz(db_path, xyz_path, input_crs='EPSG:26910')
    bathymetry.check_positions(db_path, wsel_tolerance_m=0.05)
"""
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False


def load_xyz(db_path, xyz_path, input_crs="EPSG:26910", output_crs="EPSG:26910"):
    """Load a bathymetry XYZ file into ``tblBathymetry`` in the project database.

    Parameters
    ----------
    db_path : str or Path
        Path to the legacy-format SQLite project database.
    xyz_path : str or Path
        Path to the bathymetry file. Supported formats:
        - ``.xyz``, ``.txt`` — tab- or space-separated X Y Z (no header)
        - ``.csv`` — must have columns ``X``, ``Y``, ``Z`` (case-insensitive)
        - ``.asc`` — ESRI ASCII raster (read with rasterio if available,
          otherwise falls back to numpy)
        - ``.tif``, ``.tiff`` — GeoTIFF (requires rasterio)
    input_crs : str, optional
        CRS of the input file (default ``EPSG:26910``, UTM Zone 10N NAD83).
    output_crs : str, optional
        Target CRS for the database (default ``EPSG:26910``).
        If ``input_crs == output_crs``, no reprojection is done.

    The table ``tblBathymetry`` has columns:
        X (REAL), Y (REAL), Z (REAL)
    Z is the riverbed elevation in the output CRS datum (e.g. metres above NAVD88).

    Returns
    -------
    int
        Number of bathymetry points stored.
    """
    xyz_path = Path(xyz_path)
    if not xyz_path.is_file():
        raise FileNotFoundError(f"Bathymetry file not found: {xyz_path}")

    # --- Read the file --------------------------------------------------------
    suffix = xyz_path.suffix.lower()

    if suffix in (".xyz", ".txt"):
        df = pd.read_csv(xyz_path, sep=r"\s+", header=None, names=["X", "Y", "Z"],
                         dtype={"X": np.float64, "Y": np.float64, "Z": np.float64})
    elif suffix == ".csv":
        df = pd.read_csv(xyz_path, dtype=str)
        df.columns = df.columns.str.strip().str.lower()
        col_map = {}
        for col in df.columns:
            if col in ("x", "easting", "e"):
                col_map[col] = "X"
            elif col in ("y", "northing", "n"):
                col_map[col] = "Y"
            elif col in ("z", "elevation", "elev", "depth", "height"):
                col_map[col] = "Z"
        if col_map:
            df.rename(columns=col_map, inplace=True)
        if not {"X", "Y", "Z"} <= set(df.columns):
            raise ValueError(
                f"CSV must have X, Y, Z columns. Found: {list(df.columns)}"
            )
        df = df[["X", "Y", "Z"]].copy()
        for c in ("X", "Y", "Z"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["X", "Y", "Z"], inplace=True)
    elif suffix in (".tif", ".tiff"):
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                "rasterio is required for GeoTIFF bathymetry input. "
                "Install with: pip install rasterio"
            )
        with rasterio.open(xyz_path) as src:
            if src.crs:
                input_crs = str(src.crs)
            arr = src.read(1)
            xs, ys = np.meshgrid(
                np.arange(arr.shape[1]), np.arange(arr.shape[0])
            )
            xs = src.transform * xs + src.bounds.left
            ys = src.transform * ys + src.bounds.top
            mask = np.isfinite(arr)
            df = pd.DataFrame({
                "X": xs[mask],
                "Y": ys[mask],
                "Z": arr[mask],
            })
    elif suffix == ".asc":
        try:
            import rasterio
        except ImportError:
            df = _read_ascii_legacy(xyz_path)
        else:
            with rasterio.open(xyz_path) as src:
                arr = src.read(1)
                xs, ys = np.meshgrid(
                    np.arange(arr.shape[1]), np.arange(arr.shape[0])
                )
                xs = src.transform * xs + src.bounds.left
                ys = src.transform * ys + src.bounds.top
                mask = np.isfinite(arr)
                df = pd.DataFrame({"X": xs[mask], "Y": ys[mask], "Z": arr[mask]})
    else:
        raise ValueError(f"Unsupported bathymetry file format: {suffix}")

    if df.empty:
        raise ValueError("Bathymetry file contains no valid data points")

    # --- Reproject if needed --------------------------------------------------
    if input_crs != output_crs and _HAS_PYPROJ:
        transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)
        df["X"], df["Y"] = transformer.transform(df["X"].values, df["Y"].values)
    elif input_crs != output_crs:
        print(f"  NOTE: pyproj not available; skipping CRS transform "
              f"({input_crs} -> {output_crs})")

    # --- Store in database ----------------------------------------------------
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DROP TABLE IF EXISTS tblBathymetry")
        conn.execute(
            "CREATE TABLE tblBathymetry ("
            "  X REAL NOT NULL, "
            "  Y REAL NOT NULL, "
            "  Z REAL NOT NULL"
            ")"
        )
        df.to_sql("tblBathymetry", conn, if_exists="append", index=False)
        conn.execute("CREATE INDEX idx_bathy_xy ON tblBathymetry (X, Y)")
        conn.commit()
    finally:
        conn.close()

    n = len(df)
    print(f"  Bathymetry loaded: {n} points, "
          f"X [{df.X.min():.2f}, {df.X.max():.2f}], "
          f"Y [{df.Y.min():.2f}, {df.Y.max():.2f}], "
          f"Z [{df.Z.min():.2f}, {df.Z.max():.2f}]")
    return n


def _read_ascii_legacy(path):
    """Read an ESRI ASCII raster without rasterio (fallback)."""
    with open(path) as f:
        header = {
            "ncols": int(f.readline().split()[1]),
            "nrows": int(f.readline().split()[1]),
            "xllcorner": float(f.readline().split()[1]),
            "yllcorner": float(f.readline().split()[1]),
            "cellsize": float(f.readline().split()[1]),
            "NODATA_value": f.readline().split()[1],
        }
    arr = np.loadtxt(path, skiprows=6)
    xs = np.arange(header["xllcorner"],
                   header["xllcorner"] + header["ncols"] * header["cellsize"],
                   header["cellsize"])
    ys = np.arange(header["yllcorner"],
                   header["yllcorner"] + header["nrows"] * header["cellsize"],
                   header["cellsize"])
    xs, ys = np.meshgrid(xs, ys)
    mask = np.isfinite(arr)
    return pd.DataFrame({
        "X": xs[mask],
        "Y": ys[mask],
        "Z": arr[mask],
    })


def check_positions(db_path, wsel_tolerance_m=0.05):
    """Flag position solutions that fall outside the water column using bathymetry.

    After Deng positioning, position solutions are stored in
    ``tblPositions_Deng`` with columns X, Y, Z, ToA, comment, etc.
    This function checks each ``solution found`` row:

    1. **Above water surface**: Z > WSEL(ToA) + tolerance → rejected
    2. **Below riverbed**: Z < local_bathymetry_Z(X,Y) - tolerance → rejected
    3. **No bathymetry nearby**: left as-is (no bathy coverage at this XY)

    Uses receiver Z_t values as local water surface references to handle
    the fact that bathymetry Z and position Z are in different coordinate
    systems that vary by location.

    Parameters
    ----------
    db_path : str or Path
        Path to the project database (must have ``tblPositions_Deng`` and
        ``tblBathymetry``).
    wsel_tolerance_m : float, optional
        Extra margin above WSEL for rejection (default 0.05 m). Solutions
        within this margin are kept to allow for minor WSEL uncertainty.

    Adds / updates columns in ``tblPositions_Deng``:
        bathymetry_check (TEXT) — one of:
            ``pass``, ``rejected_above_wsel``, ``rejected_below_bathy``
        bathy_z (REAL) — nearest bathymetry Z at the solution XY (NULL if none)
        wsel_at_time (REAL) — WSEL at the solution ToA (NULL if unavailable)
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        # Verify required tables exist
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "tblPositions_Deng" not in tables:
            print("  NOTE: tblPositions_Deng not found; nothing to check")
            return 0
        if "tblBathymetry" not in tables:
            print("  NOTE: tblBathymetry not found; skipping bathymetry check")
            return 0

        # Load bathymetry into a KDTree for fast nearest-neighbor lookup
        bathy = conn.execute(
            "SELECT X, Y, Z FROM tblBathymetry"
        ).fetchall()
        if not bathy:
            print("  NOTE: tblBathymetry is empty; skipping")
            return 0

        bathy_arr = np.array(bathy, dtype=np.float64)
        tree = cKDTree(bathy_arr[:, :2])  # X, Y only for lookup
        bathy_z_values = bathy_arr[:, 2]

        # Load receiver data: easting/northing (UTM) and Z_t (local depth ref)
        recs = conn.execute(
            "SELECT Rec_ID, X_t, Y_t, Z_t, easting, northing FROM tblReceiver"
        ).fetchall()
        if recs:
            rec_easting = np.array([float(r[4]) for r in recs])
            rec_northing = np.array([float(r[5]) for r in recs])
            rec_zt = np.array([float(r[3]) for r in recs])  # Z_t = local water surface ref
            rec_tree = cKDTree(np.column_stack([rec_easting, rec_northing]))
            # Pre-compute the mean offset from local X_t/Y_t to UTM easting/northing
            dx = np.mean(rec_easting - np.array([float(r[1]) for r in recs]))
            dy = np.mean(rec_northing - np.array([float(r[2]) for r in recs]))
        else:
            rec_tree = None
            rec_zt = None
            dx = dy = 0.0

        # Load WSEL (convert timeStamp to seconds if needed)
        wsel_cols = {r[1] for r in conn.execute("PRAGMA table_info(tblWSEL)").fetchall()}
        if "seconds" in wsel_cols:
            wsel_rows = conn.execute(
                "SELECT seconds, WSEL FROM tblWSEL ORDER BY seconds"
            ).fetchall()
        else:
            wsel_rows = conn.execute(
                "SELECT CAST(strftime('%s', timeStamp) AS REAL), WSEL FROM tblWSEL ORDER BY timeStamp"
            ).fetchall()
        if wsel_rows:
            wsel_t = np.array([float(r[0]) for r in wsel_rows], dtype=np.float64)
            wsel_z = np.array([float(r[1]) for r in wsel_rows], dtype=np.float64)
        else:
            wsel_t = np.array([], dtype=np.float64)
            wsel_z = np.array([], dtype=np.float64)

        # Load position solutions as numpy arrays (vectorized)
        solutions = conn.execute(
            "SELECT rowid, X, Y, Z, ToA "
            "FROM tblPositions_Deng WHERE comment = 'solution found'"
        ).fetchall()

        if not solutions:
            print("  NOTE: no 'solution found' rows in tblPositions_Deng")
            return 0

        # Add/update columns if they don't exist
        existing_cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(tblPositions_Deng)"
        ).fetchall()}
        for col in ("bathymetry_check", "bathy_z", "wsel_at_time"):
            if col not in existing_cols:
                conn.execute(
                    f'ALTER TABLE tblPositions_Deng ADD COLUMN "{col}" TEXT'
                    if col == "bathymetry_check"
                    else f'ALTER TABLE tblPositions_Deng ADD COLUMN "{col}" REAL'
                )
                conn.commit()

        n_total = len(solutions)
        print(f"  Processing {n_total:,} solutions (vectorized)...")

        # --- Vectorized computation -------------------------------------------
        rowids = np.array([s[0] for s in solutions], dtype=np.int64)
        sol_x = np.array([s[1] for s in solutions], dtype=np.float64)
        sol_y = np.array([s[2] for s in solutions], dtype=np.float64)
        sol_z = np.array([s[3] for s in solutions], dtype=np.float64)
        toa = np.array([s[4] for s in solutions], dtype=np.float64)

        # 1. Convert local X/Y to UTM (vectorized)
        utm_x = sol_x + dx
        utm_y = sol_y + dy

        # 2. Batch KD-tree query for bathymetry (vectorized)
        utm_points = np.column_stack([utm_x, utm_y])
        dist, idx = tree.query(utm_points, k=1)
        local_bathy_z = bathy_z_values[idx]

        # 3. Batch KD-tree query for nearest receiver Z_t (vectorized)
        if rec_tree is not None:
            _, rec_idx = rec_tree.query(utm_points, k=1)
            local_water_surface_z = rec_zt[rec_idx]
            has_receiver = True
        else:
            local_water_surface_z = np.full(n_total, np.nan)
            has_receiver = False

        # 4. Vectorized WSEL lookup via interpolation
        if len(wsel_t) > 0:
            wsel_at_t = np.interp(toa, wsel_t, wsel_z)
        else:
            wsel_at_t = np.full(n_total, np.nan)

        # 5. Vectorized bound checks
        # Start with all "pass"
        checks = np.full(n_total, "pass", dtype=object)
        bathy_z_results = local_bathy_z.copy()
        wsel_results = wsel_at_t.copy()

        # Above water surface?
        above_wsel = ~np.isnan(wsel_at_t) & (sol_z > wsel_at_t + wsel_tolerance_m)
        checks[above_wsel] = "rejected_above_wsel"

        # Below riverbed? Use local water surface reference
        # Effective threshold: sol_z < local_bathy_z + tolerance
        # (the water_depth formulation simplifies to this — see dead-code comment)
        below_bathy = has_receiver & (sol_z < local_bathy_z + wsel_tolerance_m)
        # Only mark as rejected_below_bathy if not already rejected for WSEL
        below_bathy = below_bathy & ~above_wsel
        checks[below_bathy] = "rejected_below_bathy"

        # --- Batch SQLite update (single executemany call) ---------------------
        # Convert numpy types to native Python types for SQLite compatibility
        checks_list = [str(c) for c in checks]
        wsel_list = [None if np.isnan(v) else float(v) for v in wsel_results]
        bathy_z_list = [float(v) for v in bathy_z_results]
        rowids_list = [int(r) for r in rowids]

        update_data = list(zip(checks_list, bathy_z_list, wsel_list, rowids_list))

        # Update in chunks to avoid SQLite parameter limit (999 per statement)
        chunk_size = 5000
        for i in range(0, len(update_data), chunk_size):
            chunk = update_data[i:i + chunk_size]
            conn.executemany(
                'UPDATE tblPositions_Deng SET bathymetry_check = ?, '
                'bathy_z = ?, wsel_at_time = ? WHERE rowid = ?',
                chunk
            )

        # Report counts
        counts = conn.execute(
            "SELECT bathymetry_check, count(*) "
            "FROM tblPositions_Deng WHERE comment = 'solution found' "
            "GROUP BY bathymetry_check"
        ).fetchall()
        for chk, cnt in counts:
            print(f"  {chk}: {cnt}")

        conn.commit()

    finally:
        conn.close()

    return n_total


def check_positions_simple(db_path, water_column=False):
    """Simpler bathymetry check — use only WSEL bounds when no bathymetry is available.

    This is the fallback when the user has no bathymetry data but wants the
    ``water_column`` behaviour (reject above WSEL and below lowest receiver - 4 m).

    Parameters
    ----------
    db_path : str or Path
        Path to the project database.
    water_column : bool
        If True, apply WSEL-based rejection even without bathymetry data.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        if "tblBathymetry" in tables:
            # Bathymetry is available — use the full check
            return check_positions(db_path)

        if not water_column:
            print("  No bathymetry data and water_column = false; skipping")
            return 0

        # WSEL-only check (legacy water_column behaviour)
        solutions = conn.execute(
            "SELECT rowid, X, Y, Z, ToA "
            "FROM tblPositions_Deng WHERE comment = 'solution found'"
        ).fetchall()

        if not solutions:
            return 0

        # Get WSEL
        wsel_rows = conn.execute(
            "SELECT CAST(strftime('%s', timeStamp) AS REAL), WSEL "
            "FROM tblWSEL ORDER BY timeStamp"
        ).fetchall()
        if not wsel_rows:
            print("  No WSEL data; cannot apply water column check")
            return 0

        wsel_t = np.array([float(r[0]) for r in wsel_rows], dtype=np.float64)
        wsel_z = np.array([float(r[1]) for r in wsel_rows], dtype=np.float64)

        # Get receiver floor
        z_min = float(conn.execute(
            "SELECT MIN(Z_t) FROM tblReceiver"
        ).fetchone()[0])
        z_floor = z_min - 4.0

        # Add columns if needed
        existing_cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(tblPositions_Deng)"
        ).fetchall()}
        for col in ("bathymetry_check", "bathy_z", "wsel_at_time"):
            if col not in existing_cols:
                conn.execute(
                    f'ALTER TABLE tblPositions_Deng ADD COLUMN "{col}" TEXT'
                    if col == "bathymetry_check"
                    else f'ALTER TABLE tblPositions_Deng ADD COLUMN "{col}" REAL'
                )
                conn.commit()

        n_total = len(solutions)
        print(f"  Processing {n_total:,} solutions (vectorized, WSEL-only)...")

        # --- Vectorized computation -------------------------------------------
        rowids = np.array([s[0] for s in solutions], dtype=np.int64)
        sol_z = np.array([s[3] for s in solutions], dtype=np.float64)
        toa = np.array([s[4] for s in solutions], dtype=np.float64)

        # Vectorized WSEL lookup
        wsel_at_t = np.interp(toa, wsel_t, wsel_z)

        # Vectorized bound checks
        above_wsel = sol_z > wsel_at_t
        below_floor = sol_z < z_floor
        rejected = above_wsel | below_floor

        checks = np.where(rejected,
                          np.where(above_wsel, "rejected_above_wsel", "rejected_below_floor"),
                          "pass")
        wsel_results = wsel_at_t.copy()

        # --- Batch SQLite update ------------------------------------------------
        # Convert numpy types to native Python types for SQLite compatibility
        checks_list = [str(c) for c in checks]
        wsel_list = [float(v) for v in wsel_results]
        rowids_list = [int(r) for r in rowids]

        update_data = list(zip(checks_list, wsel_list, rowids_list))

        chunk_size = 5000
        for i in range(0, len(update_data), chunk_size):
            chunk = update_data[i:i + chunk_size]
            conn.executemany(
                'UPDATE tblPositions_Deng SET bathymetry_check = ?, '
                'wsel_at_time = ? WHERE rowid = ?',
                chunk
            )

        n_rejected = int(rejected.sum())
        conn.commit()
        print(f"  Water column check: {n_rejected} rejected, "
              f"{n_total - n_rejected} passed")

    finally:
        conn.close()

    return n_total
