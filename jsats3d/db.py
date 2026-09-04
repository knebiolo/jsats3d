# -*- coding: utf-8 -*-
"""Database initialization, schema creation, and study parameter management for jsats3d."""

import os
import sqlite3
import logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def create_project_db(directory: str, dbName: str) -> None:
    """Create an empty, standardized SQLite database with all required schema tables.

    The schema supports raw telemetry detections, receiver metadata, metronome
    synchronization, multipath filter results, clock corrections, and 3D positioning.
    """
    db_path = os.path.join(directory, dbName)
    conn = sqlite3.connect(db_path, timeout=30.0)
    c = conn.cursor()

    # Drop existing tables if re-creating
    c.execute("DROP TABLE IF EXISTS tblTag")
    c.execute("DROP TABLE IF EXISTS tblReceiver")
    c.execute("DROP TABLE IF EXISTS tblWSEL")
    c.execute("DROP TABLE IF EXISTS tblTemp")
    c.execute("DROP TABLE IF EXISTS tblInterpolatedTemp")
    c.execute("DROP TABLE IF EXISTS tblStudyParameters")
    c.execute("DROP TABLE IF EXISTS tblDetectionRaw")
    c.execute("DROP TABLE IF EXISTS tblMetronomeUnfiltered")
    c.execute("DROP TABLE IF EXISTS tblMetronomeFiltered")
    c.execute("DROP TABLE IF EXISTS tblMetronomeSecondFiltered")
    c.execute("DROP TABLE IF EXISTS tblDetectionFilterPrimary")
    c.execute("DROP TABLE IF EXISTS tblDetectionFilterSecondary")
    c.execute("DROP TABLE IF EXISTS tblDetectionClockFixed")
    c.execute("DROP TABLE IF EXISTS tblPositions_Deng")

    # Create tables
    c.execute(
        """CREATE TABLE tblStudyParameters (
            UTC_Conv INTEGER,
            BM_Elev REAL,
            BM_Elev_Units TEXT,
            Output_Units TEXT,
            masterReceiver TEXT,
            synch_time_start TIMESTAMP,
            synch_time_end TIMESTAMP
        )"""
    )

    c.execute(
        """CREATE TABLE tblTag (
            Tag_ID TEXT PRIMARY KEY,
            TagType TEXT,
            pulseRate REAL
        )"""
    )

    c.execute(
        """CREATE TABLE tblReceiver (
            Rec_ID TEXT PRIMARY KEY,
            Tag_ID TEXT,
            X REAL,
            Y REAL,
            Z REAL,
            X_t REAL,
            Y_t REAL,
            Z_t REAL,
            Ref_Elev TEXT
        )"""
    )

    c.execute(
        """CREATE TABLE tblWSEL (
            timeStamp TIMESTAMP,
            WSEL REAL
        )"""
    )

    c.execute(
        """CREATE TABLE tblInterpolatedTemp (
            timeStamp TIMESTAMP,
            C REAL
        )"""
    )

    c.execute(
        """CREATE TABLE tblDetectionRaw (
            Tag_ID TEXT,
            Rec_ID TEXT,
            timeStamp TIMESTAMP,
            seconds REAL,
            FreqOff REAL,
            Amplitude REAL,
            NBW REAL,
            SNR REAL,
            Valid INTEGER,
            Pascals REAL,
            Celsius REAL
        )"""
    )

    c.execute(
        """CREATE TABLE tblMetronomeUnfiltered (
            Sequence INTEGER,
            Year INTEGER,
            Month INTEGER,
            Day INTEGER,
            Hour INTEGER,
            Minute INTEGER,
            Second INTEGER,
            UnixSeconds REAL,
            Microseconds REAL,
            Tag_ID TEXT,
            FreqOff REAL,
            Amplitude REAL,
            NBW REAL,
            SNR REAL,
            Valid INTEGER,
            Pascals REAL,
            Celsius REAL,
            Rec_ID TEXT,
            timeStamp TIMESTAMP,
            seconds REAL,
            metronome_transmission REAL
        )"""
    )

    c.execute(
        """CREATE TABLE tblMetronomeFiltered (
            Tag_ID TEXT,
            Rec_ID TEXT,
            timeStamp TIMESTAMP,
            seconds REAL,
            Amplitude REAL,
            NBW REAL,
            SNR REAL,
            metronome_transmission REAL,
            det_rank REAL,
            multipath INTEGER
        )"""
    )

    c.execute(
        """CREATE TABLE tblMetronomeSecondFiltered (
            Tag_ID TEXT,
            Rec_ID TEXT,
            timeStamp TIMESTAMP,
            seconds REAL,
            Amplitude REAL,
            NBW REAL,
            SNR REAL,
            transNo REAL,
            multipath_prediction INTEGER
        )"""
    )

    c.execute(
        """CREATE TABLE tblDetectionFilterPrimary (
            Tag_ID TEXT,
            Rec_ID TEXT,
            timeStamp TIMESTAMP,
            seconds_fix REAL,
            transNo REAL,
            Amplitude REAL,
            NBW REAL,
            SNR REAL,
            multipath INTEGER
        )"""
    )

    c.execute(
        """CREATE TABLE tblDetectionFilterSecondary (
            Tag_ID TEXT,
            Rec_ID TEXT,
            timeStamp TIMESTAMP,
            seconds_fix REAL,
            transNo REAL,
            Amplitude REAL,
            NBW REAL,
            SNR REAL,
            multipath INTEGER,
            multipath_prediction INTEGER
        )"""
    )

    c.execute(
        """CREATE TABLE tblDetectionClockFixed (
            Tag_ID TEXT,
            Rec_ID TEXT,
            timeStamp TIMESTAMP,
            seconds REAL,
            seconds_fix REAL,
            timeDiff REAL,
            avg_C REAL,
            SoS REAL,
            Amplitude REAL,
            NBW REAL,
            SNR REAL
        )"""
    )

    c.execute(
        """CREATE TABLE tblPositions_Deng (
            transNo REAL,
            solNo REAL,
            r0 TEXT,
            r1 TEXT,
            r2 TEXT,
            r3 TEXT,
            X REAL,
            Y REAL,
            Z REAL,
            T01 REAL,
            ToA REAL,
            comment TEXT,
            in_hull INTEGER,
            solution TEXT,
            Tag_ID TEXT
        )"""
    )

    conn.commit()
    c.close()
    conn.close()
    logger.info("Successfully created project database schema in %s", db_path)


def set_study_parameters(
    utc_conv: int,
    bm_elev: float,
    bm_elev_units: str,
    output_units: str,
    masterReceiver: str,
    synch_time_start,
    synch_time_end,
    dbName: str,
) -> None:
    """Set study parameters in the project database."""
    conn = sqlite3.connect(dbName, timeout=30.0)
    c = conn.cursor()
    params = [(
        utc_conv,
        bm_elev,
        bm_elev_units,
        output_units,
        masterReceiver,
        synch_time_start,
        synch_time_end,
    )]
    c.execute("DROP TABLE IF EXISTS tblStudyParameters")
    c.execute(
        """CREATE TABLE tblStudyParameters (
            UTC_Conv INTEGER,
            BM_Elev REAL,
            BM_Elev_Units TEXT,
            Output_Units TEXT,
            masterReceiver TEXT,
            synch_time_start TIMESTAMP,
            synch_time_end TIMESTAMP
        )"""
    )

    c.executemany("INSERT INTO tblStudyParameters VALUES (?,?,?,?,?,?,?)", params)
    conn.commit()
    c.close()
    conn.close()
    logger.info("Study parameters configured in %s", dbName)


def temp_interpolator(projectDB: str, interp_type: str):
    """Create a temperature interpolator function over time (in unix seconds).

    interp_type describes the interpolator type, either 'cubic' or 'linear'.
    """
    conn = sqlite3.connect(projectDB, timeout=30.0)
    temp = pd.read_sql("SELECT * FROM tblInterpolatedTemp", con=conn)
    conn.close()

    temp["timeStamp"] = pd.to_datetime(temp.timeStamp)
    temp.sort_values("timeStamp", inplace=True)

    # Convert timestamps to numeric seconds compatible with pandas >= 2.0
    seconds = pd.to_datetime(temp.timeStamp).astype("int64") / 1.0e9
    temp["seconds"] = seconds.values
    temp.drop_duplicates("seconds", keep="first", inplace=True)
    temp.set_index("seconds", inplace=True, drop=False)

    interpolator = interp1d(
        temp.seconds.values,
        temp.C.values,
        kind=interp_type,
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interpolator


def avg_temp(row, temp_interpolator):
    """Evaluate average temperature across multiple interpolators for a given row."""
    seconds = row[1]["seconds"]
    temps = []
    for i in temp_interpolator:
        temps.append(temp_interpolator[i](seconds))
    return np.nanmean(temps)
