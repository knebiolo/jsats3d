# -*- coding: utf-8 -*-
"""Kernel density utilization distribution calculations and 3D volume visualization."""

import os
import sqlite3
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class kernels:
    """Python class to construct 3D kernel utilization distributions for telemetry positions."""

    def __init__(self, pos_type: str, projectDB: str, outputWS: str, tag_ID: str = None):
        self.projectDB = projectDB
        self.tag_ID = tag_ID
        self.outputWS = outputWS

        if pos_type == "Deng":
            conn = sqlite3.connect(projectDB, timeout=30.0)
            if tag_ID is not None:
                sql = "SELECT * FROM tblPositions_Deng WHERE Tag_ID = ?"
                self.dat = pd.read_sql(sql, con=conn, params=[tag_ID])
                self.dat = self.dat[(self.dat.comment == "solution found") & (self.dat.solution == "B")]
            else:
                sql = "SELECT * FROM tblPositions_Deng"
                self.dat = pd.read_sql(sql, con=conn)
                self.dat = self.dat[(self.dat.comment == "solution found") & (self.dat.solution == "B")]
                self.dat = self.dat.iloc[::10, :]

            recSQL = "SELECT * FROM tblReceiver"
            self.ephemeris = pd.read_sql(recSQL, con=conn)
            self.ephemeris.set_index("Rec_ID", drop=False, inplace=True)
            conn.close()

            pos = self.dat[["X", "Y", "Z"]].iloc[::5, :]
            X = pos.X.values
            Y = pos.Y.values
            Z = pos.Z.values
            pos_arr = np.vstack([X, Y, Z])

            self.dat["datetime"] = pd.to_datetime(self.dat.ToA, errors="coerce")
            self.dat["hour"] = self.dat["datetime"].dt.hour
            self.dat["daytime"] = self.dat["hour"].apply(lambda x: True if 6 <= x < 18 else False)

            logger.info("Length of position array: %d records", len(pos))

            self.xmin = self.ephemeris.X_t.min()
            self.ymin = self.ephemeris.Y_t.min()
            self.zmin = self.ephemeris.Z_t.min()
            self.xmax = self.ephemeris.X_t.max()
            self.ymax = self.ephemeris.Y_t.max()
            self.zmax = self.ephemeris.Z_t.max()

            d = pos_arr.shape[0]
            n = pos_arr.shape[1]
            bw = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4.0)) if n > 0 else 1.0

            logger.info("Fitting kernel density estimate with bandwidth %f", bw)
            self.kde = KernelDensity(bandwidth=bw, metric="euclidean", kernel="gaussian", algorithm="ball_tree")
            if pos_arr.shape[1] > 0:
                self.kde.fit(pos_arr.T)

            xi, yi, zi = np.mgrid[
                self.xmin:self.xmax:50j,
                self.ymin:self.ymax:50j,
                self.zmin:self.zmax:50j,
            ]
            self.coords = np.vstack([xi.ravel(), yi.ravel(), zi.ravel()])
            if pos_arr.shape[1] > 0:
                self.density = np.reshape(np.exp(self.kde.score_samples(self.coords.T)), xi.shape)
                max_dens = self.density.flatten().max()
                self.density_norm = self.density / max_dens if max_dens > 0 else self.density
            else:
                self.density_norm = np.zeros(xi.shape)

            logger.info("Kernel density estimation complete.")

    def plot(self):
        xi, yi, zi = np.mgrid[
            self.xmin:self.xmax:50j,
            self.ymin:self.ymax:50j,
            self.zmin:self.zmax:50j,
        ]
        self.fig = go.Figure(
            data=go.Volume(
                x=xi.flatten(),
                y=yi.flatten(),
                z=zi.flatten(),
                value=self.density_norm.flatten(),
                isomin=0.025,
                opacity=0.1,
                surface_count=15,
            )
        )
        out_file = os.path.join(self.outputWS, f"{self.tag_ID}.html")
        self.fig.write_html(out_file, auto_open=False)
        logger.info("Saved 3D volume plot to %s", out_file)
