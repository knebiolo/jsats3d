# -*- coding: utf-8 -*-
"""Unit tests for temperature interpolator generation in jsats3d."""

import sqlite3
import numpy as np
import pandas as pd
from jsats3d.db import create_project_db, temp_interpolator


def test_temp_interpolator_synthetic_data(tmp_path):
    """Verify temperature interpolator behavior on timestamped temperature data."""
    db_name = "test_temp.db"
    create_project_db(str(tmp_path), db_name)
    db_path = str(tmp_path / db_name)

    # Insert synthetic temperature data spanning 1 hour (timestamps in ISO format)
    timestamps = [
        "2026-06-01 00:00:00",
        "2026-06-01 00:30:00",
        "2026-06-01 01:00:00",
    ]
    temps = [15.0, 17.0, 19.0]
    temp_df = pd.DataFrame({"timeStamp": timestamps, "C": temps})

    conn = sqlite3.connect(db_path)
    temp_df.to_sql("tblInterpolatedTemp", con=conn, if_exists="append", index=False)
    conn.close()

    interpolator = temp_interpolator(db_path, "linear")

    # Convert timestamps to unix seconds
    sec_start = pd.to_datetime("2026-06-01 00:00:00").timestamp()
    sec_mid = pd.to_datetime("2026-06-01 00:15:00").timestamp()
    sec_end = pd.to_datetime("2026-06-01 01:00:00").timestamp()

    assert np.isclose(interpolator(sec_start), 15.0)
    assert np.isclose(interpolator(sec_mid), 16.0)
    assert np.isclose(interpolator(sec_end), 19.0)
