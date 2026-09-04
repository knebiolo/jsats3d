# -*- coding: utf-8 -*-
"""Beacon transmission enumeration, metronome alignment, and receiver clock drift correction."""

import os
import sqlite3
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from .db import temp_interpolator
from .positioning import sos

logger = logging.getLogger(__name__)


class beacon_epoch:
    """Python class to identify transmission epochs for beacon tags across host and child receivers."""

    def __init__(self, tag: str, projectDB: str, scratchWS: str):
        self.tag = tag
        self.projectDB = projectDB
        self.scratchWS = scratchWS

        conn = sqlite3.connect(projectDB, timeout=30.0)

        # Get tag's pulse rate and type
        tagSQL = "SELECT pulseRate, TagType FROM tblTag WHERE Tag_ID = ?"
        tagDat = pd.read_sql(tagSQL, con=conn, params=[tag])
        self.pulseRate = tagDat.pulseRate.values[0]
        self.tagType = tagDat.TagType.values[0]

        rec_df = pd.read_sql("SELECT Rec_ID FROM tblReceiver WHERE Tag_ID = ?", con=conn, params=[tag])
        receiver = rec_df.Rec_ID.values[0]

        # Get host and child receiver data
        host_dat_SQL = "SELECT * FROM tblDetectionRaw WHERE Tag_ID = ? AND Rec_ID = ?"
        self.host_dat = pd.read_sql(host_dat_SQL, con=conn, params=[tag, receiver])

        child_dat_SQL = "SELECT * FROM tblDetectionRaw WHERE Tag_ID = ? AND Rec_ID != ?"
        self.child_dat = pd.read_sql(child_dat_SQL, con=conn, params=[tag, receiver])
        conn.close()

        # Build MultiIndexes
        if not self.host_dat.empty:
            i_arrays = [self.host_dat.Rec_ID.values, self.host_dat.Tag_ID.values, self.host_dat.seconds.values]
            i_tuples = list(zip(*i_arrays))
            i_index = pd.MultiIndex.from_tuples(i_tuples, names=["Rec_ID", "Tag_ID", "seconds"])
            self.host_dat.set_index(i_index, inplace=True)
            self.host_dat.sort_index(level="seconds", inplace=True)
            self.host_dat.drop_duplicates(keep="first", inplace=True)

        if not self.child_dat.empty:
            j_arrays = [self.child_dat.Rec_ID.values, self.child_dat.Tag_ID.values, self.child_dat.seconds.values]
            j_tuples = list(zip(*j_arrays))
            j_index = pd.MultiIndex.from_tuples(j_tuples, names=["Rec_ID", "Tag_ID", "seconds"])
            self.child_dat.set_index(j_index, inplace=True)
            self.child_dat.sort_index(level="seconds", inplace=True)
            self.child_dat.drop_duplicates(keep="first", inplace=True)

    def host_receiver_enumeration(self):
        if self.host_dat.empty:
            logger.warning("Host data is empty for beacon tag %s", self.tag)
            return

        self.host_dat["lag"] = self.host_dat.seconds.diff()
        self.host_dat["lag"].fillna(0, inplace=True)
        self.host_dat["metronome_transmission"] = np.nan
        transNo = 0

        for i in self.host_dat.iterrows():
            curr_lag = i[1]["lag"]
            if curr_lag < 0.5 * self.pulseRate:
                self.host_dat.at[i[0], "metronome_transmission"] = transNo
            else:
                transNo += 1
                self.host_dat.at[i[0], "metronome_transmission"] = transNo

        conn = sqlite3.connect(self.projectDB, timeout=30.0)
        c = conn.cursor()
        self.host_dat.dropna(axis=0, subset=["metronome_transmission"], inplace=True)
        self.host_dat.to_sql("tblMetronomeUnfiltered", conn, if_exists="append", index=False)
        c.close()
        conn.close()

    def adjacent_receiver_enumeration(self):
        if self.child_dat.empty or self.host_dat.empty:
            return

        self.child_dat["metronome_transmission"] = 0.0
        for i in self.host_dat.metronome_transmission.values:
            trans_time = self.host_dat[self.host_dat.metronome_transmission == i].seconds.min()
            dl = trans_time - (0.5 * self.pulseRate)
            ul = trans_time + (0.5 * self.pulseRate)
            self.child_dat.loc[
                (self.child_dat.seconds >= dl) & (self.child_dat.seconds <= ul),
                "metronome_transmission",
            ] = i

        conn = sqlite3.connect(self.projectDB, timeout=30.0)
        self.child_dat.to_sql("tblMetronomeUnfiltered", conn, if_exists="append", index=False)
        conn.commit()
        conn.close()

    def indexer(self):
        conn = sqlite3.connect(self.projectDB, timeout=30.0)
        c = conn.cursor()
        try:
            c.execute(
                "CREATE INDEX idx_combined_metronome_unfiltered ON tblMetronomeUnfiltered (Rec_ID, Tag_ID, seconds)"
            )
            conn.commit()
        except sqlite3.OperationalError as e:
            logger.debug("Index creation note: %s", e)
        finally:
            c.close()
            conn.close()


class clock_fix_object:
    """Class holding data and metadata required for clock drift estimation."""

    def __init__(self, curr_receiver: str, receiver_list: list, projectDB: str, scratchWS: str, figureWS: str):
        self.current_receiver = curr_receiver
        self.receiver_list = receiver_list
        self.projectDB = projectDB
        self.scratchWS = scratchWS
        self.figureWS = figureWS

        conn = sqlite3.connect(projectDB, timeout=30.0)

        # Get receiver metadata
        curr_rec_dat = pd.read_sql("SELECT * FROM tblReceiver WHERE Rec_ID = ?", con=conn, params=[curr_receiver])
        self.current_tag_id = curr_rec_dat.at[0, "Tag_ID"]
        self.ref_elev = curr_rec_dat.Ref_Elev.values[0]

        # Get pulse rate of current tag
        self.current_pulse_rate = pd.read_sql(
            "SELECT pulseRate FROM tblTag WHERE Tag_ID = ?", con=conn, params=[self.current_tag_id]
        ).pulseRate.values[0]

        # Get receiver coordinates
        placeholders = ",".join(["?"] * len(receiver_list))
        recSQL = f"SELECT * FROM tblReceiver WHERE Rec_ID IN ({placeholders})"
        self.receivers = pd.read_sql(recSQL, con=conn, params=list(receiver_list))
        self.receivers.set_index("Rec_ID", drop=False, inplace=True)

        self.master_clock_rec_ID = pd.read_sql(
            "SELECT masterReceiver FROM tblStudyParameters", con=conn
        ).masterReceiver.values[0]

        self.master_clock_tag_ID = self.receivers[
            self.receivers.Rec_ID == self.master_clock_rec_ID
        ].Tag_ID.values[0]

        self.master_pulse_rate = pd.read_sql(
            "SELECT pulseRate FROM tblTag WHERE Tag_ID = ?", con=conn, params=[self.master_clock_tag_ID]
        ).pulseRate.values[0]

        self.ref_elev = self.receivers[self.receivers.Rec_ID == self.master_clock_rec_ID].Ref_Elev.values[0]

        # Get transmission timestamps for master clock
        sql = """SELECT transNo, seconds FROM tblMetronomeFiltered
                 WHERE Rec_ID = ? AND Tag_ID = ? AND multipath == 0"""
        self.ToT = pd.read_sql_query(sql, con=conn, params=[self.master_clock_rec_ID, self.master_clock_tag_ID])
        self.ToT.rename(columns={"seconds": "ToT"}, inplace=True)
        self.ToT.set_index("transNo", inplace=True)
        self.ToT.drop_duplicates(inplace=True)

        # Get WSEL data
        WSELdf = pd.read_sql("SELECT * FROM tblWSEL", con=conn)
        WSELdf["timeStamp"] = pd.to_datetime(WSELdf.timeStamp)
        WSELdf["seconds"] = pd.to_datetime(WSELdf.timeStamp).astype("int64") / 1.0e9

        self.benchmark_elev = pd.read_sql("SELECT BM_Elev FROM tblStudyParameters", con=conn).values[0][0]
        self.elev_units = pd.read_sql("SELECT BM_Elev_Units FROM tblStudyParameters", con=conn).BM_Elev_Units.values[0]
        self.output_units = pd.read_sql("SELECT Output_Units FROM tblStudyParameters", con=conn).Output_Units.values[0]

        if self.elev_units == "feet" and self.output_units == "meters":
            WSELdf["WSEL"] = WSELdf.WSEL / 3.28084
            self.benchmark_elev = self.benchmark_elev / 3.28084

        self.WSELfun = interp1d(WSELdf.seconds, WSELdf.WSEL, kind="linear")

        # Get filtered recapture data
        dataSQL = """SELECT * FROM tblMetronomeSecondFiltered
                     WHERE Rec_ID = ? AND Tag_ID = ? AND multipath_prediction == 0"""
        self.clock_data = pd.read_sql(dataSQL, con=conn, params=[self.current_receiver, self.master_clock_tag_ID])
        self.clock_data["lag"] = self.clock_data.seconds.diff()
        self.clock_data["leap"] = np.abs(self.clock_data.seconds.diff(-1))

        conn.close()


def clock_fix(clock_fix_obj: clock_fix_object):
    """Fix clocks on a receiver-by-receiver basis."""
    master_receiver = clock_fix_obj.master_clock_rec_ID
    current_receiver = clock_fix_obj.current_receiver
    master_elev_ref = clock_fix_obj.receivers[
        clock_fix_obj.receivers.Rec_ID == master_receiver
    ].Ref_Elev.values[0]

    receiver_dat = clock_fix_obj.clock_data.copy()
    logger.info("Length of clock data for %s: %d rows", current_receiver, len(receiver_dat))

    if receiver_dat.empty:
        logger.warning("No data for receiver %s, check inputs", current_receiver)
        return

    x1 = clock_fix_obj.receivers[clock_fix_obj.receivers.Rec_ID == master_receiver].X_t.values[0]
    y1 = clock_fix_obj.receivers[clock_fix_obj.receivers.Rec_ID == master_receiver].Y_t.values[0]
    z1 = clock_fix_obj.receivers[clock_fix_obj.receivers.Rec_ID == master_receiver].Z_t.values[0]
    x2 = clock_fix_obj.receivers[clock_fix_obj.receivers.Rec_ID == current_receiver].X_t.values[0]
    y2 = clock_fix_obj.receivers[clock_fix_obj.receivers.Rec_ID == current_receiver].Y_t.values[0]
    z2 = clock_fix_obj.receivers[clock_fix_obj.receivers.Rec_ID == current_receiver].Z_t.values[0]

    receiver_dat["x1"] = x1
    receiver_dat["y1"] = y1
    if master_elev_ref == "BM":
        receiver_dat["z1"] = z1
    else:
        t = receiver_dat.seconds.values
        Zt = clock_fix_obj.WSELfun(t) - z1
        z1_fix = clock_fix_obj.benchmark_elev - Zt
        receiver_dat["z1"] = z1_fix

    receiver_dat["x2"] = x2
    receiver_dat["y2"] = y2
    if clock_fix_obj.ref_elev == "BM":
        receiver_dat["z2"] = z2
    else:
        t = receiver_dat.seconds.values
        Zt = clock_fix_obj.WSELfun(t) - z2
        z2_fix = clock_fix_obj.benchmark_elev - Zt
        receiver_dat["z2"] = z2_fix

    def dist_fun(row):
        from_pos = np.array([row["x1"], row["y1"], row["z1"]])
        to_pos = np.array([row["x2"], row["y2"], row["z2"]])
        return np.linalg.norm(to_pos - from_pos)

    receiver_dat["dist"] = receiver_dat.apply(dist_fun, axis=1)
    receiver_dat.sort_values(by="seconds", ascending=True, inplace=True)

    if receiver_dat.empty:
        return

    receiver_dat = receiver_dat[receiver_dat.transNo != 0]
    receiver_dat.set_index("transNo", inplace=True)
    receiver_dat = receiver_dat.join(clock_fix_obj.ToT, how="left")
    receiver_dat.dropna(axis=0, how="any", subset=["ToT", "seconds"], inplace=True)
    receiver_dat.reset_index(inplace=True)

    if receiver_dat.empty:
        logger.warning("After join, receiver_dat is empty for %s", current_receiver)
        return

    interpolator = temp_interpolator(clock_fix_obj.projectDB, "linear")
    conn = sqlite3.connect(clock_fix_obj.projectDB, timeout=30.0)
    temp = pd.read_sql("SELECT * FROM tblInterpolatedTemp", con=conn)
    conn.close()

    temp["timeStamp"] = pd.to_datetime(temp.timeStamp)
    temp.sort_values("timeStamp", inplace=True)
    seconds = pd.to_datetime(temp.timeStamp).astype("int64") / 1.0e9
    temp["seconds"] = seconds.values

    receiver_dat["TDoA"] = receiver_dat.seconds - receiver_dat.ToT
    receiver_dat["TDoAlag"] = receiver_dat.TDoA.diff()

    if current_receiver != master_receiver:
        times = receiver_dat.seconds.values
        avg_C = interpolator(times)
        SoS = sos(avg_C)
        receiver_dat["avg_C"] = avg_C
        receiver_dat["SoS"] = SoS

        receiver_dat["ToA_expected"] = receiver_dat.ToT + receiver_dat.dist / receiver_dat.SoS
        receiver_dat["TDoA_expected"] = receiver_dat.dist / receiver_dat.SoS
        receiver_dat["ToA_error"] = receiver_dat.seconds - receiver_dat.ToA_expected
        receiver_dat["TDoA_error"] = receiver_dat.TDoA - receiver_dat.TDoA_expected
        receiver_dat["seconds_fix"] = receiver_dat.seconds - receiver_dat.ToA_error
        receiver_dat["TDoA_t"] = receiver_dat.TDoA - receiver_dat.TDoA_error
        receiver_dat["DDoA"] = receiver_dat.SoS * receiver_dat.TDoA

        # DBSCAN filtering loop
        from sklearn.cluster import DBSCAN

        iters = 5
        for i in range(iters):
            t_min = receiver_dat.seconds.min()
            t_max = receiver_dat.seconds.max()
            if pd.isna(t_min) or pd.isna(t_max) or t_min == t_max:
                break

            ts = np.arange(t_min, t_max, clock_fix_obj.master_pulse_rate)
            if len(ts) < 2:
                break

            f = interp1d(
                receiver_dat.seconds,
                receiver_dat.DDoA,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            ddoa = f(ts)

            euc_dist = np.sqrt(np.power(np.diff(ts), 2) + np.power(np.diff(ddoa), 2))
            if len(euc_dist) == 0:
                break
            ext_dist = pd.DataFrame(euc_dist).quantile(q=[0.90, 0.95, 0.96, 0.97, 0.98, 0.99]).values

            eps_val = ext_dist[0][0] if i == 0 else ext_dist[1][0]
            if eps_val <= 0:
                eps_val = 1e-3

            test = DBSCAN(eps=eps_val, min_samples=3, metric="euclidean").fit(np.vstack((ts, ddoa)).T)
            results = pd.DataFrame.from_dict({"ts": ts, "ddoa": ddoa, "class_": test.labels_})

            good_bad = np.where(results.class_ < 0, 0, 1)
            multi_pred = interp1d(ts, good_bad, kind="next", bounds_error=False, fill_value="extrapolate")

            receiver_dat["DBSCAN_multi"] = multi_pred(receiver_dat.seconds)
            receiver_dat = receiver_dat[receiver_dat["DBSCAN_multi"] == 1]

        receiver_dat["deltaDDoA"] = receiver_dat.DDoA.diff()
        clock_fix_obj.receiver_dat = receiver_dat

        # Plot output
        try:
            gs = gridspec.GridSpec(4, 4)
            gs.update(hspace=0.5)
            fig = plt.figure(figsize=(6, 7))
            ax1 = plt.subplot(gs[0:2, :])
            ax1.plot(receiver_dat.transNo.values, receiver_dat.DDoA.values, "ro", label="recorded")
            ax1.set_xlabel("Transmission Number")
            ax1.set_ylabel("Distance Difference of Arrival (m)")
            ax1.set_title(
                f"Effects of Clock Drift: \n Difference between ToA at Receiver {current_receiver} and ToT from Receiver {master_receiver}"
            )

            ax2 = plt.subplot(gs[2:4, :])
            ax2.plot(clock_fix_obj.receivers.X_t.values, clock_fix_obj.receivers.Y_t.values, "bo", label="Receiver")
            ax2.set_xlabel("Easting (m)")
            ax2.set_ylabel("Northing (m)")
            ax2.legend(title="Stationary Receivers")
            plt.savefig(
                os.path.join(
                    clock_fix_obj.figureWS,
                    f"ClockDrift_f{master_receiver}_t{current_receiver}.png",
                ),
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as e:
            logger.debug("Plot generation skipped/failed: %s", e)

        receiver_dat.sort_values(by="seconds", inplace=True)
        residual = interp1d(
            receiver_dat.seconds.values,
            receiver_dat.ToA_error.values,
            kind="linear",
            bounds_error=False,
            fill_value=9999,
        )
        receiver_dat["seconds_residual_predicted"] = residual(receiver_dat.seconds)
        receiver_dat["prev_seconds"] = receiver_dat.seconds.shift()
        receiver_dat["prev_error"] = receiver_dat.ToA_error.shift()
        receiver_dat["beta1"] = receiver_dat.prev_error
        receiver_dat["beta2"] = (receiver_dat.ToA_error - receiver_dat.prev_error) / (
            receiver_dat.seconds - receiver_dat.prev_seconds
        )

        conn = sqlite3.connect(clock_fix_obj.projectDB, timeout=30.0)
        sql = "SELECT * FROM tblDetectionRaw WHERE Rec_ID = ?"
        curr_rec_dat = pd.read_sql_query(sql, conn, params=[current_receiver])
        conn.close()

        curr_rec_dat["timeDiff"] = 0.0
        curr_rec_dat.sort_values(by="seconds", inplace=True)
        curr_rec_dat["seconds_residual"] = residual(curr_rec_dat.seconds)
        curr_rec_dat = curr_rec_dat[curr_rec_dat.seconds_residual != 9999]
        curr_rec_dat["seconds_fix"] = curr_rec_dat.seconds - curr_rec_dat.seconds_residual

        if not curr_rec_dat.empty:
            firstDet = curr_rec_dat.seconds_fix.min()
            curr_rec_dat["timeDiff"] = curr_rec_dat.seconds_fix.values - firstDet
            times = curr_rec_dat.seconds_fix.values
            avg_C = interpolator(times)
            SoS = sos(avg_C)
            curr_rec_dat["avg_C"] = avg_C
            curr_rec_dat["SoS"] = SoS

        curr_rec_dat.to_csv(
            os.path.join(clock_fix_obj.scratchWS, f"receiver_{current_receiver}_epoch_fix.csv"),
            index=False,
            float_format="%.6f",
        )
        receiver_dat.to_csv(
            os.path.join(clock_fix_obj.figureWS, f"receiver_{current_receiver}_clock_fix.csv"),
            float_format="%.6f",
        )

    else:
        conn = sqlite3.connect(clock_fix_obj.projectDB, timeout=30.0)
        sql = "SELECT * FROM tblDetectionRaw WHERE Rec_ID = ?"
        curr_rec_dat = pd.read_sql_query(sql, conn, params=[current_receiver])
        conn.close()

        curr_rec_dat["timeDiff"] = 0.0
        curr_rec_dat.sort_values(by="seconds", inplace=True)
        curr_rec_dat["seconds_residual"] = 0.0
        curr_rec_dat["seconds_fix"] = curr_rec_dat.seconds - curr_rec_dat.seconds_residual

        if not curr_rec_dat.empty:
            times = curr_rec_dat.seconds_fix.values
            avg_C = interpolator(times)
            SoS = sos(avg_C)
            curr_rec_dat["avg_C"] = avg_C
            curr_rec_dat["SoS"] = SoS

        curr_rec_dat.to_csv(
            os.path.join(clock_fix_obj.scratchWS, f"receiver_{current_receiver}_epoch_fix.csv"),
            index=False,
            float_format="%.6f",
        )
        receiver_dat.to_csv(
            os.path.join(clock_fix_obj.figureWS, f"receiver_{current_receiver}_clock_fix.csv"),
            float_format="%.6f",
        )


def epoch_fix_data_management(inputWS: str, projectDB: str):
    """Aggregate receiver epoch fix CSVs and insert into tblDetectionClockFixed."""
    files = [f for f in os.listdir(inputWS) if os.path.isfile(os.path.join(inputWS, f))]
    conn = sqlite3.connect(projectDB, timeout=30.0)
    for f in files:
        file_path = os.path.join(inputWS, f)
        dat = pd.read_csv(file_path)
        dat.to_sql("tblDetectionClockFixed", con=conn, index=False, if_exists="append", chunksize=1000)
        os.remove(file_path)
    conn.commit()
    conn.close()
    logger.info("Imported epoch fix data from %d files into tblDetectionClockFixed", len(files))
