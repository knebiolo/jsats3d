# -*- coding: utf-8 -*-
"""Synthetic geometry unit test for the Deng 3D TDOA positioning solver."""

import sqlite3
import numpy as np
import pandas as pd
from jsats3d.db import create_project_db
from jsats3d.positioning import position, sos


def test_deng_tdoa_solver_synthetic_geometry(tmp_path):
    """Verify that Deng 3D TDOA positioning recovers a known source position within tolerance."""
    db_name = "test_deng.db"
    create_project_db(str(tmp_path), db_name)
    db_path = str(tmp_path / db_name)
    out_dir = str(tmp_path)

    conn = sqlite3.connect(db_path)

    # Insert study parameters
    study_params = pd.DataFrame([{
        "UTC_Conv": 0,
        "BM_Elev": 100.0,
        "BM_Elev_Units": "feet",
        "Output_Units": "feet",
        "masterReceiver": "REC1",
        "synch_time_start": "2026-01-01 00:00:00",
        "synch_time_end": "2026-01-02 00:00:00",
    }])
    study_params.to_sql("tblStudyParameters", con=conn, if_exists="append", index=False)

    # Insert tag metadata
    tag_df = pd.DataFrame([{
        "Tag_ID": "TAG1",
        "TagType": "study",
        "pulseRate": 5.0,
    }])
    tag_df.to_sql("tblTag", con=conn, if_exists="append", index=False)

    # Known receiver coordinates (in feet)
    receivers = [
        {"Rec_ID": "REC1", "Tag_ID": "BEACON1", "X": 0.0, "Y": 0.0, "Z": 0.0, "X_t": 0.0, "Y_t": 0.0, "Z_t": 0.0, "Ref_Elev": "BM"},
        {"Rec_ID": "REC2", "Tag_ID": "BEACON2", "X": 100.0, "Y": 0.0, "Z": 0.0, "X_t": 100.0, "Y_t": 0.0, "Z_t": 0.0, "Ref_Elev": "BM"},
        {"Rec_ID": "REC3", "Tag_ID": "BEACON3", "X": 0.0, "Y": 100.0, "Z": 0.0, "X_t": 0.0, "Y_t": 100.0, "Z_t": 0.0, "Ref_Elev": "BM"},
        {"Rec_ID": "REC4", "Tag_ID": "BEACON4", "X": 0.0, "Y": 0.0, "Z": 50.0, "X_t": 0.0, "Y_t": 0.0, "Z_t": 50.0, "Ref_Elev": "BM"},
    ]
    pd.DataFrame(receivers).to_sql("tblReceiver", con=conn, if_exists="append", index=False)

    # Temperature data (20.0 degrees C)
    temp_df = pd.DataFrame([{
        "timeStamp": "2026-01-01 00:00:00",
        "C": 20.0,
    }, {
        "timeStamp": "2026-01-02 00:00:00",
        "C": 20.0,
    }])
    temp_df.to_sql("tblInterpolatedTemp", con=conn, if_exists="append", index=False)

    # Water surface elevation
    wsel_df = pd.DataFrame([{
        "timeStamp": "2026-01-01 00:00:00",
        "WSEL": 100.0,
    }, {
        "timeStamp": "2026-01-02 00:00:00",
        "WSEL": 100.0,
    }])
    wsel_df.to_sql("tblWSEL", con=conn, if_exists="append", index=False)

    # Known source position S = (25.0, 30.0, 10.0)
    source = np.array([25.0, 30.0, 10.0])
    tot = 1000.000000
    temp_c = 20.0
    c_speed = sos(temp_c)

    detections = []
    for rec in receivers:
        rec_pos = np.array([rec["X_t"], rec["Y_t"], rec["Z_t"]])
        dist = np.linalg.norm(rec_pos - source)
        toa = tot + (dist / c_speed)
        detections.append({
            "Tag_ID": "TAG1",
            "Rec_ID": rec["Rec_ID"],
            "timeStamp": "2026-01-01 00:00:00",
            "seconds_fix": toa,
            "transNo": 1.0,
            "Amplitude": 100.0,
            "NBW": 10.0,
            "SNR": 20.0,
            "multipath": 0,
            "multipath_prediction": 0,
        })

    pd.DataFrame(detections).to_sql("tblDetectionFilterSecondary", con=conn, if_exists="append", index=False)
    conn.close()

    # Instantiate positioning object and solve Deng TDOA
    pos_obj = position(
        tag="TAG1",
        resolved_clocks=["REC1", "REC2", "REC3", "REC4"],
        projectDB=db_path,
        outputWS=out_dir,
        figureWS=out_dir,
    )
    pos_obj.Deng()

    # Check solutions
    sol_df_a = pos_obj.DengSolutionA_unfiltered
    sol_df_b = pos_obj.DengSolutionB_unfiltered

    valid_solutions = pd.concat([
        sol_df_a[sol_df_a.comment == "solution found"],
        sol_df_b[sol_df_b.comment == "solution found"],
    ])

    assert len(valid_solutions) > 0, "No valid positioning solutions found"

    recovered_x = valid_solutions.iloc[0]["X"]
    recovered_y = valid_solutions.iloc[0]["Y"]
    recovered_z = valid_solutions.iloc[0]["Z"]

    assert np.isclose(recovered_x, source[0], atol=0.1)
    assert np.isclose(recovered_y, source[1], atol=0.1)
    assert np.isclose(recovered_z, source[2], atol=0.1)
