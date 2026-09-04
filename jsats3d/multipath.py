# -*- coding: utf-8 -*-
"""Multipath detection, feature extraction, classifier training, and prediction."""

import os
import sqlite3
import logging
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing, tree
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class multipath_data_object:
    """Class to hold detection data for multipath feature extraction."""

    def __init__(self, tag: str, projectDB: str, scratchWS: str, metronome: bool = False):
        self.tag = tag
        self.metronome = metronome
        self.projectDB = projectDB
        self.scratchWS = scratchWS

        conn = sqlite3.connect(projectDB, timeout=30.0)

        # Query tag metadata
        tagSQL = "SELECT pulseRate, TagType FROM tblTag WHERE Tag_ID = ?"
        tagDat = pd.read_sql(tagSQL, con=conn, params=[tag])
        self.pulseRate = tagDat.at[0, "pulseRate"]
        self.tagType = tagDat.at[0, "TagType"]

        param_df = pd.read_sql("SELECT masterReceiver FROM tblStudyParameters", con=conn)
        self.master_receiver = param_df.masterReceiver.values[0]

        if metronome:
            datSQL = "SELECT * FROM tblMetronomeUnfiltered WHERE Tag_ID = ?"
        else:
            datSQL = "SELECT * FROM tblDetectionClockFixed WHERE Tag_ID = ?"

        self.data = pd.read_sql(datSQL, con=conn, params=[tag])
        conn.close()

        if len(self.data) > 0:
            self.empty = False
            time_col = "seconds" if metronome else "seconds_fix"
            i_arrays = [self.data.Rec_ID.values, self.data.Tag_ID.values, self.data[time_col].values]
            i_tuples = list(zip(*i_arrays))
            index = pd.MultiIndex.from_tuples(i_tuples, names=["Rec_ID", "Tag_ID", time_col])
            self.data.set_index(index, inplace=True, drop=True)

            if not metronome:
                if self.tagType == "study":
                    firstDet = self.data.seconds_fix.min()
                    epochDiff = self.data.seconds_fix.values - firstDet
                    transNo = np.round(epochDiff / self.pulseRate, 0)
                    self.data["transNo"] = transNo
                else:
                    self.data["transNo"] = np.nan
                    transNo = 0
                    self.data.reset_index(drop=True, inplace=True)
                    det_counts = (
                        self.data.groupby(["Rec_ID"])["seconds_fix"]
                        .count()
                        .to_frame()
                        .rename(columns={"seconds_fix": "row_count"})
                    )
                    det_counts.reset_index(drop=False, inplace=True)
                    self.data.set_index(index, inplace=True, drop=True)
                    max_count = det_counts.row_count.max()
                    host_rec = det_counts[det_counts.row_count == max_count].Rec_ID.values[0]

                    host_dat = self.data[self.data.Rec_ID == host_rec].copy()
                    host_dat["lag"] = host_dat.seconds_fix.diff()
                    host_dat["lag"].fillna(0, inplace=True)

                    for i in host_dat.iterrows():
                        curr_lag = i[1]["lag"]
                        if curr_lag < 0.5 * self.pulseRate:
                            host_dat.at[i[0], "transNo"] = transNo
                            self.data.at[i[0], "transNo"] = transNo
                        else:
                            transNo += 1
                            host_dat.at[i[0], "transNo"] = transNo
                            self.data.at[i[0], "transNo"] = transNo

                    for i in host_dat.transNo.values:
                        trans_time = host_dat[host_dat.transNo == i].seconds_fix.min()
                        dl = trans_time - (0.5 * self.pulseRate)
                        ul = trans_time + (0.5 * self.pulseRate)
                        self.data.loc[
                            (self.data.seconds_fix >= dl) & (self.data.seconds_fix <= ul),
                            "transNo",
                        ] = i

            self.data.drop_duplicates(keep="first", inplace=True)
        else:
            self.empty = True


def multipath_2(multipath_object: multipath_data_object):
    """Rank detection epochs per receiver to identify multipath detections."""
    if multipath_object.empty:
        return

    receivers = np.unique(multipath_object.data.Rec_ID.values)
    for i in receivers:
        recDat = multipath_object.data[multipath_object.data.Rec_ID == i].copy()
        if not multipath_object.metronome:
            recDat.sort_index(level="seconds_fix", inplace=True)
            grouped = recDat.groupby(by=["transNo"])["seconds_fix"].rank().rename("det_rank")
            recDat = recDat.join(grouped, how="left")

            transNo = recDat.transNo.values
            seconds_fix = recDat.seconds_fix.values
            i_arrays = [transNo, seconds_fix]
            i_tuples = list(zip(*i_arrays))
            index = pd.MultiIndex.from_tuples(i_tuples, names=["transNo", "seconds_fix"])
            recDat.set_index(index, inplace=True)
        else:
            recDat.sort_index(level="seconds", inplace=True)
            grouped = recDat.groupby(by=["metronome_transmission"])["seconds"].rank().rename("det_rank")
            recDat = recDat.join(grouped, how="left")
            recDat.rename(columns={"metronome_transmission": "transNo"}, inplace=True)
            if "lag" in recDat.columns:
                recDat.drop("lag", axis=1, inplace=True)

            transNo = recDat.transNo.values
            seconds = recDat.seconds.values
            i_arrays = [transNo, seconds]
            i_tuples = list(zip(*i_arrays))
            index = pd.MultiIndex.from_tuples(i_tuples, names=["transNo", "seconds"])
            recDat.set_index(index, inplace=True)

        recDat.reset_index(drop=True, inplace=True)
        recDat["multipath"] = np.where(recDat.det_rank.values > 1.0, 1, 0)

        if not multipath_object.metronome:
            recDat.to_csv(
                os.path.join(
                    multipath_object.scratchWS,
                    f"{multipath_object.tag}_{i}_multipath.csv",
                ),
                index=False,
                float_format="%.6f",
            )
        else:
            conn = sqlite3.connect(multipath_object.projectDB, timeout=30.0)
            recDat.to_sql("tblMetronomeFiltered", conn, if_exists="append", index=False)
            conn.close()

    if multipath_object.metronome:
        conn = sqlite3.connect(multipath_object.projectDB, timeout=30.0)
        conn.commit()
        conn.close()


def multipath_data_management(
    inputWS: str, projectDB: str, primary: bool = True, metronome: bool = False
):
    """Import multipath CSVs into project database."""
    if metronome:
        tblName = "tblMetronomeSecondFiltered"
    elif primary:
        tblName = "tblDetectionFilterPrimary"
    else:
        tblName = "tblDetectionFilterSecondary"

    files = [f for f in os.listdir(inputWS) if os.path.isfile(os.path.join(inputWS, f))]
    conn = sqlite3.connect(projectDB, timeout=30.0)
    for f in files:
        file_path = os.path.join(inputWS, f)
        dat = pd.read_csv(file_path)
        dat.to_sql(tblName, con=conn, index=False, if_exists="append", chunksize=1000)
        os.remove(file_path)

    conn.commit()
    conn.close()


def multipath_classifier(
    tag: str,
    projectDB: str,
    outputWS: str,
    metronome: bool = False,
    method: str = "SVM",
):
    """Train and run supervised/unsupervised machine learning models to filter multipath detections."""
    conn = sqlite3.connect(projectDB, timeout=30.0)
    if not metronome:
        dat = pd.read_sql(
            "SELECT * FROM tblDetectionFilterPrimary WHERE Tag_ID = ?",
            con=conn,
            params=[tag],
        )
    else:
        dat = pd.read_sql(
            "SELECT * FROM tblMetronomeFiltered WHERE Tag_ID = ?",
            con=conn,
            params=[tag],
        )
        rec_ID = pd.read_sql(
            "SELECT Rec_ID FROM tblReceiver WHERE Tag_ID = ?",
            con=conn,
            params=[tag],
        ).Rec_ID.values[0]
    conn.close()

    dat = dat[dat.SNR > 0].copy()
    if dat.empty:
        logger.warning("No detections with SNR > 0 for tag %s", tag)
        return

    recs = dat.Rec_ID.unique()
    for i in recs:
        rec_dat = dat[dat.Rec_ID == i].copy()
        rec_dat.set_index("transNo", inplace=True, drop=False)
        rec_dat.sort_index(inplace=True)

        if metronome:
            rec_dat = rec_dat[rec_dat.Rec_ID != rec_ID]

        multi = rec_dat[rec_dat.multipath == 1].copy()
        primary = rec_dat[rec_dat.multipath == 0].copy()

        if len(rec_dat) > 100 and len(multi) > 0:
            primary[["amp_n", "nbw_n", "snr_n"]] = preprocessing.normalize(
                primary[["Amplitude", "NBW", "SNR"]]
            )
            multi[["amp_n", "nbw_n", "snr_n"]] = preprocessing.normalize(
                multi[["Amplitude", "NBW", "SNR"]]
            )

            if len(rec_dat) > 5000:
                primary_filtered_chunks = []
                chunks = np.arange(0, 10, 1)
                primary["chunk"] = np.random.choice(chunks, len(primary))
                for chunk in chunks:
                    chunk_df = primary[primary.chunk == chunk].copy()
                    if chunk_df.empty:
                        continue

                    neighbors = NearestNeighbors(n_neighbors=min(20, len(chunk_df)))
                    features = np.vstack((chunk_df.amp_n, chunk_df.nbw_n, chunk_df.snr_n)).T
                    neighbors_fit = neighbors.fit(features)
                    distances, _ = neighbors_fit.kneighbors(features)
                    distances = np.sort(distances, axis=0)

                    ext_dist = np.quantile(distances, q=[0.1, 0.4, 0.75, 0.80, 0.90, 0.95, 0.99])
                    eps_val = ext_dist[5] if len(ext_dist) > 5 else 0.1

                    test = DBSCAN(eps=eps_val, min_samples=6, metric="euclidean").fit(features)
                    chunk_df["dbscan_cluster"] = test.labels_

                    chunk_df = chunk_df[chunk_df.dbscan_cluster != -1]
                    if not chunk_df.empty:
                        clst_cnt = chunk_df.groupby(["dbscan_cluster"])["NBW"].min()
                        chunk_df = chunk_df[chunk_df.dbscan_cluster == clst_cnt.idxmin()]
                        chunk_df.drop(["dbscan_cluster"], axis=1, inplace=True)
                        primary_filtered_chunks.append(chunk_df)

                primary_filtered = (
                    pd.concat(primary_filtered_chunks, ignore_index=True)
                    if primary_filtered_chunks
                    else primary
                )
            else:
                primary_filtered = primary

            train_dat = pd.concat([primary_filtered, multi], ignore_index=True)
            logger.info("Generated training and testing datasets for receiver %s", i)

            X = train_dat[["Amplitude", "NBW", "SNR"]]
            X_scaled = preprocessing.scale(X)
            X_norm = preprocessing.normalize(X)

            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                X_scaled, train_dat[["multipath"]], test_size=0.3, random_state=111
            )
            X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
                X_norm, train_dat[["multipath"]], test_size=0.3, random_state=111
            )

            if method == "SVM":
                svc = SVC(kernel="rbf", C=1e10)
                svc.fit(X_train_s, y_train_s.values.ravel())
                y_pred = svc.predict(X_test_s)
                classifier = svc
            elif method == "NB":
                nb = GaussianNB()
                nb.fit(X_train_s, y_train_s.values.ravel())
                y_pred = nb.predict(X_test_s)
                classifier = nb
            elif method == "CART":
                tre = tree.DecisionTreeClassifier()
                tre.fit(X_train_n, y_train_n.values.ravel())
                y_pred = tre.predict(X_test_n)
                classifier = tre
            elif method == "KNN":
                knn = KNeighborsClassifier(n_neighbors=2)
                knn.fit(X_train_s, y_train_s.values.ravel())
                y_pred = knn.predict(X_test_s)
                classifier = knn
            else:
                raise ValueError("Invalid algorithm choice; must be SVM, NB, KNN, or CART")

            target_test = y_test_n if method == "CART" else y_test_s
            logger.info(
                "Accuracy for %s (%s): %f",
                i,
                method,
                metrics.accuracy_score(target_test, y_pred),
            )

            if method in ["SVM", "NB", "KNN"]:
                rec_dat[["amp_s", "nbw_s", "snr_s"]] = preprocessing.scale(
                    rec_dat[["Amplitude", "NBW", "SNR"]]
                )
                rec_dat["multipath_prediction"] = classifier.predict(
                    rec_dat[["amp_s", "nbw_s", "snr_s"]]
                )
            else:
                rec_dat[["amp_n", "nbw_n", "snr_n"]] = preprocessing.normalize(
                    rec_dat[["Amplitude", "NBW", "SNR"]]
                )
                rec_dat["multipath_prediction"] = classifier.predict(
                    rec_dat[["amp_n", "nbw_n", "snr_n"]]
                )

        else:
            if len(rec_dat) > 1:
                rec_dat[["amp_n", "nbw_n", "snr_n"]] = preprocessing.normalize(
                    rec_dat[["Amplitude", "NBW", "SNR"]]
                )
                euc_dist = np.sqrt(
                    np.power(np.diff(rec_dat.Amplitude), 2)
                    + np.power(np.diff(rec_dat.NBW), 2)
                    + np.power(np.diff(rec_dat.SNR), 2)
                )
                if len(euc_dist) > 0:
                    ext_dist = np.quantile(euc_dist, [0.90, 0.95, 0.99])
                    eps_val = np.abs(ext_dist[0])
                    if eps_val <= 0:
                        eps_val = 1e-3

                    min_s = max(1, len(primary) // 2)
                    features = np.vstack(
                        (rec_dat.Amplitude, rec_dat.NBW, rec_dat.SNR)
                    ).T
                    test = DBSCAN(
                        eps=eps_val, min_samples=min_s, metric="euclidean"
                    ).fit(features)
                    rec_dat["multipath_prediction"] = test.labels_
                    rec_dat = rec_dat[rec_dat["multipath_prediction"] != -1].copy()
                else:
                    rec_dat["multipath_prediction"] = 0
            else:
                rec_dat["multipath_prediction"] = 0

        output_file = os.path.join(outputWS, f"multipath_predict_{tag}_at_{i}.csv")
        rec_dat.to_csv(output_file, index=False, float_format="%.6f")
