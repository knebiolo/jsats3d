# -*- coding: utf-8 -*-
"""Data ingestion routines for raw cabled receivers and study metadata into jsats3d project databases."""

import os
import sqlite3
import logging
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def study_data_import(dataFrame: pd.DataFrame, dbName: str, tblName: str) -> None:
    """Import formatted study data (e.g. tblTag or tblReceiver) into project database."""
    conn = sqlite3.connect(dbName, timeout=30.0)
    dataFrame.to_sql(tblName, con=conn, index=False, if_exists="append")
    conn.commit()
    conn.close()
    logger.info("Imported %d records into table %s in %s", len(dataFrame), tblName, dbName)


def teknologic_import(UTC_conv: int, inputWS: str, dbName: str, recName: str) -> None:
    """Import raw detection data from Teknologic cabled receivers into tblDetectionRaw."""
    def parse_time_stamp(row):
        return datetime(
            int(row["Year"]),
            int(row["Month"]),
            int(row["Day"]),
            int(row["Hour"]),
            int(row["Minute"]),
            int(row["Second"]),
        )

    conn = sqlite3.connect(dbName, timeout=30.0)
    synch_df = pd.read_sql("SELECT synch_time_start, synch_time_end FROM tblStudyParameters", con=conn)
    synch_time_start = pd.to_datetime(synch_df.synch_time_start.values[0])
    synch_time_end = pd.to_datetime(synch_df.synch_time_end.values[0])

    files = [f for f in os.listdir(inputWS) if os.path.isfile(os.path.join(inputWS, f))]
    det_list = []

    for f in files:
        detFile = os.path.join(inputWS, f)
        det = pd.read_csv(
            detFile,
            header=None,
            names=[
                "Sequence",
                "Year",
                "Month",
                "Day",
                "Hour",
                "Minute",
                "Second",
                "UnixSeconds",
                "Microseconds",
                "Tag_ID",
                "FreqOff",
                "Amplitude",
                "NBW",
                "SNR",
                "Valid",
                "Pascals",
                "Celsius",
            ],
        )
        det["Rec_ID"] = recName
        det["Tag_ID"] = det["Tag_ID"].astype(str).str.strip()
        det["timeStamp"] = det.apply(parse_time_stamp, axis=1)

        # Convert timestamps to numeric seconds safely in pandas >= 2.0
        ts_index = pd.to_datetime(det.timeStamp.values)
        seconds_arr = ts_index.astype("int64") // 10**9
        det["seconds"] = seconds_arr.astype(np.float64)
        det["seconds"] = np.round(det["seconds"] + (det["Microseconds"] / 1.0e6), 6)

        logger.info(
            "File %s: length before synchronization filter: %d", f, len(det)
        )
        det = det[(det.timeStamp > synch_time_start) & (det.timeStamp < synch_time_end)]
        logger.info(
            "File %s: length after synchronization filter: %d", f, len(det)
        )
        det.sort_values(by="seconds", ascending=True, inplace=True)
        det_list.append(det)

    if det_list:
        det_data = pd.concat(det_list, ignore_index=True)
        det_data.drop(
            columns=[
                "Sequence",
                "Year",
                "Month",
                "Day",
                "Hour",
                "Minute",
                "Second",
                "UnixSeconds",
                "Microseconds",
            ],
            inplace=True,
            errors="ignore",
        )
        det_data.drop_duplicates(keep="first", inplace=True)
        det_data.to_sql("tblDetectionRaw", conn, if_exists="append", index=False)
        conn.commit()

    conn.close()
    logger.info("Teknologic data import completed for receiver %s", recName)


def acoustic_data_import(site: str, recType: str, rawDataFiles: str, projectDB: str) -> None:
    """Import raw acoustic telemetry data from cabled receivers."""
    conn = sqlite3.connect(projectDB, timeout=30.0)
    utc_df = pd.read_sql("SELECT UTC_Conv FROM tblStudyParameters", con=conn)
    conn.close()
    UTC_conv = utc_df.UTC_Conv.values[0]

    if recType == "Teknologic":
        teknologic_import(UTC_conv, rawDataFiles, projectDB, site)
    else:
        logger.warning("Unsupported receiver type: %s", recType)
