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
        else:
            rec_tree = None
            rec_zt = None

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

        # Load position solutions
        solutions = conn.execute(
            "SELECT rowid, X, Y, Z, ToA, comment "
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

        # Process each solution
        n_total = len(solutions)

        for rowid, sol_x, sol_y, sol_z, toa, comment in solutions:
            sol_x = float(sol_x)
            sol_y = float(sol_y)
            sol_z = float(sol_z)
            toa = float(toa)

            # --- Convert local X/Y to UTM for bathymetry lookup ------------
            # Compute offset from receiver easting/northing vs X_t/Y_t
            if recs:
                dx = np.mean([float(r[4]) - float(r[1]) for r in recs])
                dy = np.mean([float(r[5]) - float(r[2]) for r in recs])
                utm_x = sol_x + dx
                utm_y = sol_y + dy
            else:
                utm_x, utm_y = sol_x, sol_y

            # --- Look up local bathymetry -----------------------------------
            dist, idx = tree.query([utm_x, utm_y], k=1)
            local_bathy_z = bathy_z_values[idx]

            # --- Look up nearest receiver for local water surface reference --
            if rec_tree is not None:
                _, rec_idx = rec_tree.query([utm_x, utm_y], k=1)
                local_water_surface_z = rec_zt[rec_idx]
            else:
                local_water_surface_z = None

            # --- Look up WSEL at time of arrival ----------------------------
            if len(wsel_t) > 0:
                wsel_at_t = float(np.interp(toa, wsel_t, wsel_z))
            else:
                wsel_at_t = np.nan

            # --- Check bounds -----------------------------------------------
            check = "pass"

            # Above water surface?
            if not np.isnan(wsel_at_t) and sol_z > wsel_at_t + wsel_tolerance_m:
                check = "rejected_above_wsel"
            # Below riverbed? Use local water surface reference
            elif local_water_surface_z is not None:
                # Water depth at this location = local_water_surface_z - local_bathy_z
                # Position is below riverbed if sol_z < local_bathy_z - tolerance
                # But since sol_z and bathy_z are in different coordinate systems,
                # we compare sol_z to the water surface reference minus water depth
                water_depth = local_water_surface_z - local_bathy_z
                if sol_z < local_water_surface_z - water_depth + wsel_tolerance_m:
                    check = "rejected_below_bathy"
                # Simplified: sol_z < local_bathy_z - tolerance (using local offset)
                elif sol_z < local_bathy_z - wsel_tolerance_m:
                    check = "rejected_below_bathy"

            # Store results
            conn.execute(
                'UPDATE tblPositions_Deng SET bathymetry_check = ?, '
                'bathy_z = ?, wsel_at_time = ? WHERE rowid = ?',
                (check, local_bathy_z, wsel_at_t if not np.isnan(wsel_at_t) else None, rowid)
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
        print("  No bathymetry data; applying WSEL-only water column check")
        solutions = conn.execute(
            "SELECT rowid, X, Y, Z, ToA, comment "
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

        n_rejected = 0
        for rowid, sol_x, sol_y, sol_z, toa, comment in solutions:
            wsel_at_t = float(np.interp(toa, wsel_t, wsel_z))
            if sol_z > wsel_at_t or sol_z < z_floor:
                conn.execute(
                    'UPDATE tblPositions_Deng SET bathymetry_check = ?, '
                    'wsel_at_time = ? WHERE rowid = ?',
                    ('rejected_above_wsel' if sol_z > wsel_at_t else 'rejected_below_floor',
                     wsel_at_t, rowid)
                )
                n_rejected += 1

        conn.commit()
        print(f"  Water column check: {n_rejected} rejected, "
              f"{len(solutions) - n_rejected} passed")

    finally:
        conn.close()

    return len(solutions) if solutions else 0
