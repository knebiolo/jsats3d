# -*- coding: utf-8 -*-
"""Unit tests for SQLite database schema creation and study parameters in jsats3d."""

import os
import sqlite3
import pandas as pd
from jsats3d.db import create_project_db, set_study_parameters


def test_create_project_db(tmp_path):
    """Verify that create_project_db creates all required database tables."""
    db_name = "test_project.db"
    create_project_db(str(tmp_path), db_name)

    db_path = tmp_path / db_name
    assert os.path.exists(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    expected_tables = [
        "tblTag",
        "tblReceiver",
        "tblWSEL",
        "tblInterpolatedTemp",
        "tblStudyParameters",
        "tblDetectionRaw",
        "tblMetronomeUnfiltered",
        "tblMetronomeFiltered",
        "tblMetronomeSecondFiltered",
        "tblDetectionFilterPrimary",
        "tblDetectionFilterSecondary",
        "tblDetectionClockFixed",
        "tblPositions_Deng",
    ]

    for table in expected_tables:
        assert table in tables, f"Expected table '{table}' not found in created database"


def test_set_study_parameters(tmp_path):
    """Verify set_study_parameters inserts study metadata correctly."""
    db_name = "test_params.db"
    create_project_db(str(tmp_path), db_name)
    db_path = str(tmp_path / db_name)

    set_study_parameters(
        utc_conv=-5,
        bm_elev=100.0,
        bm_elev_units="feet",
        output_units="meters",
        masterReceiver="REC_01",
        synch_time_start="2026-01-01 00:00:00",
        synch_time_end="2026-01-02 00:00:00",
        dbName=db_path,
    )

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM tblStudyParameters", con=conn)
    conn.close()

    assert len(df) == 1
    assert df.at[0, "UTC_Conv"] == -5
    assert df.at[0, "masterReceiver"] == "REC_01"
    assert df.at[0, "BM_Elev"] == 100.0
