# -*- coding: utf-8 -*-
"""Speed of sound interpolation, Daniel Deng 3D TDOA exact solution solver, and positioning management."""

import os
import sqlite3
import logging
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from .db import temp_interpolator

logger = logging.getLogger(__name__)


def sos(temp: float) -> float:
    """Calculate speed of sound (ft/s) in water given temperature in Celsius."""
    arr_temp = np.arange(4, 30.5, 0.5)
    sps = np.array([
        1421.62, 1423.9, 1426.15, 1428.38, 1430.58, 1432.75, 1434.9, 1437.02,
        1439.12, 1441.19, 1443.23, 1445.25, 1447.25, 1449.22, 1451.17, 1453.09,
        1454.99, 1456.87, 1458.72, 1460.55, 1462.36, 1464.14, 1465.91, 1467.65,
        1469.36, 1471.06, 1472.73, 1474.38, 1476.01, 1477.62, 1479.21, 1480.77,
        1482.32, 1483.84, 1485.35, 1486.83, 1488.29, 1489.74, 1491.16, 1492.56,
        1493.95, 1495.32, 1496.66, 1497.99, 1499.3, 1500.59, 1501.86, 1503.11,
        1504.35, 1505.56, 1506.76, 1507.94, 1509.1
    ])  # m/s
    sps = sps * 3.2804  # convert to ft/s
    f = interp1d(arr_temp, sps, kind="cubic")
    return float(f(temp))


def sos_apply(row: pd.Series) -> float:
    """Row-wise speed of sound helper based on Celsius column."""
    return sos(row["Celsius"])


class position:
    """Acoustic positioning object using Daniel Deng's exact 3D TDOA solver."""

    def __init__(self, tag: str, resolved_clocks: list, projectDB: str, outputWS: str, figureWS: str):
        self.tag = tag
        self.resolved_clocks = resolved_clocks
        self.projectDB = projectDB
        self.outputWS = outputWS
        self.figureWS = figureWS

        conn = sqlite3.connect(projectDB, timeout=30.0)

        tag_data_list = []
        for i in self.resolved_clocks:
            sql = """SELECT * FROM tblDetectionFilterSecondary
                     WHERE Tag_ID = ? AND Rec_ID = ?"""
            dat = pd.read_sql(sql, con=conn, params=[self.tag, i])
            dat = dat[(dat.multipath != 1) & (dat.multipath_prediction != 1)]
            if not dat.empty:
                tag_data_list.append(dat)

        self.tag_data = pd.concat(tag_data_list, ignore_index=True) if tag_data_list else pd.DataFrame()

        # Build receiver ephemeris
        placeholders = ",".join(["?"] * len(resolved_clocks))
        recSQL = f"SELECT * FROM tblReceiver WHERE Rec_ID IN ({placeholders})"
        self.ephemeris = pd.read_sql(recSQL, con=conn, params=list(resolved_clocks))
        self.ephemeris.set_index("Rec_ID", drop=False, inplace=True)
        self.convex_hull = ConvexHull(np.array(self.ephemeris[["X_t", "Y_t", "Z_t"]]))

        # Get WSEL data
        WSELdf = pd.read_sql("SELECT * FROM tblWSEL", con=conn)
        WSELdf.dropna(inplace=True)
        WSELdf["timeStamp"] = pd.to_datetime(WSELdf.timeStamp)
        WSELdf["seconds"] = pd.to_datetime(WSELdf.timeStamp).astype("int64") / 1.0e9

        self.benchmark_elev = pd.read_sql("SELECT BM_Elev FROM tblStudyParameters", con=conn).values[0][0]
        self.elev_units = pd.read_sql("SELECT BM_Elev_Units FROM tblStudyParameters", con=conn).BM_Elev_Units.values[0]
        self.output_units = pd.read_sql("SELECT Output_Units FROM tblStudyParameters", con=conn).Output_Units.values[0]

        WSELdf["WSEL"] = WSELdf.WSEL / 3.28084
        self.WSELfun = interp1d(WSELdf.seconds, WSELdf.WSEL, kind="linear")

        self.tagType = pd.read_sql(
            "SELECT TagType FROM tblTag WHERE Tag_ID = ?", con=conn, params=[tag]
        ).TagType.values[0]
        self.pulseRate = pd.read_sql(
            "SELECT pulseRate FROM tblTag WHERE Tag_ID = ?", con=conn, params=[tag]
        ).pulseRate.values[0]

        self.interpolator = temp_interpolator(self.projectDB, "linear")
        conn.close()

    def Deng(self, print_output: bool = False):
        def point_in_hull(point, hull):
            tolerance = 1e-12
            return all(
                (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
                for eq in hull.equations
            )

        Solution_Cols = [
            "transNo",
            "solNo",
            "r0",
            "r1",
            "r2",
            "r3",
            "X",
            "Y",
            "Z",
            "T01",
            "ToA",
            "comment",
            "in_hull",
        ]

        sol_a_list = []
        sol_b_list = []

        if self.tag_data.empty:
            logger.warning("tag_data is empty for positioning tag %s", self.tag)
            self.DengSolutionA_unfiltered = pd.DataFrame(columns=Solution_Cols)
            self.DengSolutionB_unfiltered = pd.DataFrame(columns=Solution_Cols)
            return

        self.tag_data.sort_values(by="seconds_fix", axis=0, ascending=True, inplace=True)
        tSteps = self.tag_data.transNo.unique()

        for j in sorted(tSteps):
            tDat = self.tag_data[self.tag_data.transNo == j].copy()
            if len(tDat) >= 4:
                tDat.sort_values(by="seconds_fix", axis=0, ascending=True, inplace=True)
                tDat["rank"] = tDat.seconds_fix.rank()
                tDat.set_index("Rec_ID", drop=False, inplace=True)
                tested = []

                for combo in list(combinations(tDat.Rec_ID.values, 4)):
                    row1 = tDat[tDat.Rec_ID == combo[0]]
                    row2 = tDat[tDat.Rec_ID == combo[1]]
                    row3 = tDat[tDat.Rec_ID == combo[2]]
                    row4 = tDat[tDat.Rec_ID == combo[3]]

                    sub_dat = pd.concat([row1, row2, row3, row4])
                    sub_dat.sort_values(by="seconds_fix", axis=0, ascending=True, inplace=True)
                    recs = tuple(sub_dat.Rec_ID.values)
                    sol_no = 0

                    if recs not in tested:
                        ref = sub_dat.iloc[0].Rec_ID
                        r1 = sub_dat.iloc[1].Rec_ID
                        r2 = sub_dat.iloc[2].Rec_ID
                        r3 = sub_dat.iloc[3].Rec_ID

                        t_ref = sub_dat.seconds_fix.iloc[0]
                        t1 = sub_dat.seconds_fix.iloc[1]
                        t2 = sub_dat.seconds_fix.iloc[2]
                        t3 = sub_dat.seconds_fix.iloc[3]

                        def z_at_t(t_val, Rec_ID):
                            return self.ephemeris[self.ephemeris.Rec_ID == Rec_ID].Z_t.values[0]

                        r0Pos = np.array([
                            self.ephemeris[self.ephemeris.Rec_ID == ref].X_t.values[0],
                            self.ephemeris[self.ephemeris.Rec_ID == ref].Y_t.values[0],
                            z_at_t(t_ref, ref),
                        ])
                        r1Pos = np.array([
                            self.ephemeris[self.ephemeris.Rec_ID == r1].X_t.values[0],
                            self.ephemeris[self.ephemeris.Rec_ID == r1].Y_t.values[0],
                            z_at_t(t1, r1),
                        ])
                        r2Pos = np.array([
                            self.ephemeris[self.ephemeris.Rec_ID == r2].X_t.values[0],
                            self.ephemeris[self.ephemeris.Rec_ID == r2].Y_t.values[0],
                            z_at_t(t2, r2),
                        ])
                        r3Pos = np.array([
                            self.ephemeris[self.ephemeris.Rec_ID == r3].X_t.values[0],
                            self.ephemeris[self.ephemeris.Rec_ID == r3].Y_t.values[0],
                            z_at_t(t3, r3),
                        ])

                        R = np.array([
                            [r1Pos[0] - r0Pos[0], r2Pos[0] - r0Pos[0], r3Pos[0] - r0Pos[0]],
                            [r1Pos[1] - r0Pos[1], r2Pos[1] - r0Pos[1], r3Pos[1] - r0Pos[1]],
                            [r1Pos[2] - r0Pos[2], r2Pos[2] - r0Pos[2], r3Pos[2] - r0Pos[2]],
                        ])

                        tdoa_1 = np.round(t1 - t_ref, 6)
                        tdoa_2 = np.round(t2 - t_ref, 6)
                        tdoa_3 = np.round(t3 - t_ref, 6)
                        t_mat = np.array([[tdoa_1], [tdoa_2], [tdoa_3]])

                        avg_C = self.interpolator(t_ref)
                        SoS = sos(avg_C)

                        b1 = np.linalg.norm(r1Pos - r0Pos) ** 2 - (SoS**2 * tdoa_1**2)
                        b2 = np.linalg.norm(r2Pos - r0Pos) ** 2 - (SoS**2 * tdoa_2**2)
                        b3 = np.linalg.norm(r3Pos - r0Pos) ** 2 - (SoS**2 * tdoa_3**2)
                        b = np.array([[b1], [b2], [b3]])

                        try:
                            R_inv = np.linalg.inv(R)
                            R_T_inv = np.linalg.inv(R.T)

                            a = float((SoS**4 * (t_mat.T @ R_inv @ R_T_inv @ t_mat) - SoS**2).item())
                            p = float((-0.5 * SoS**2 * (t_mat.T @ R_inv @ R_T_inv @ b)).item())
                            q = float((0.25 * (b.T @ R_inv @ R_T_inv @ b)).item())

                            disc = p**2 - a * q
                            if disc < 0:
                                raise ValueError("Negative discriminant in quadratic solver")

                            T_0a = (-p + np.sqrt(disc)) / a
                            T_0b = (-p - np.sqrt(disc)) / a

                            if np.sign(T_0a) > 0:
                                S1a = R_inv.T @ (0.5 * b - SoS**2 * t_mat * T_0a)
                                point_a = np.array([
                                    r0Pos[0] + float(S1a[0, 0]),
                                    r0Pos[1] + float(S1a[1, 0]),
                                    r0Pos[2] + float(S1a[2, 0]),
                                ])
                                in_hull_a = point_in_hull(point_a, self.convex_hull)
                                sol_a_list.append([
                                    j, sol_no, ref, r1, r2, r3,
                                    point_a[0], point_a[1], point_a[2],
                                    T_0a, tDat.seconds_fix.values[0],
                                    "solution found", in_hull_a
                                ])
                            else:
                                sol_a_list.append([
                                    j, sol_no, ref, r1, r2, r3,
                                    9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
                                    "negative time of arrival - no solution", ""
                                ])

                            if np.sign(T_0b) > 0:
                                S1b = R_inv.T @ (0.5 * b - SoS**2 * t_mat * T_0b)
                                point_b = np.array([
                                    r0Pos[0] + float(S1b[0, 0]),
                                    r0Pos[1] + float(S1b[1, 0]),
                                    r0Pos[2] + float(S1b[2, 0]),
                                ])
                                in_hull_b = point_in_hull(point_b, self.convex_hull)
                                sol_b_list.append([
                                    j, sol_no, ref, r1, r2, r3,
                                    point_b[0], point_b[1], point_b[2],
                                    T_0b, tDat.seconds_fix.values[0],
                                    "solution found", in_hull_b
                                ])
                            else:
                                sol_b_list.append([
                                    j, sol_no, ref, r1, r2, r3,
                                    9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
                                    "negative time of arrival - no solution", ""
                                ])

                        except Exception as e:
                            logger.debug("Deng solver exception at transNo %s: %s", j, e)
                            sol_a_list.append([
                                j, sol_no, ref, r1, r2, r3,
                                9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
                                "singular matrix encountered - no solution", ""
                            ])

                        sol_no += 1
                        tested.append(recs)
            else:
                fallback_row = [
                    j, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
                    9999.0, 9999.0, 9999.0, 9999.0, 9999.0,
                    "not enough receivers for solution", ""
                ]
                sol_a_list.append(fallback_row)
                sol_b_list.append(fallback_row)

        SolutionA = pd.DataFrame(sol_a_list, columns=Solution_Cols)
        SolutionB = pd.DataFrame(sol_b_list, columns=Solution_Cols)

        for col in ["transNo", "X", "Y", "Z", "ToA"]:
            SolutionA[col] = pd.to_numeric(SolutionA[col], errors="coerce")
            SolutionB[col] = pd.to_numeric(SolutionB[col], errors="coerce")

        self.DengSolutionA_unfiltered = SolutionA
        self.DengSolutionB_unfiltered = SolutionB

        SolutionA.to_csv(os.path.join(self.outputWS, f"{self.tag}_solutionA.csv"), index=False)
        SolutionB.to_csv(os.path.join(self.outputWS, f"{self.tag}_solutionB.csv"), index=False)

    def trajectory_plot_Deng(self, hull_filter: bool = False, beacon: bool = False):
        solA = self.DengSolutionA_unfiltered[
            self.DengSolutionA_unfiltered.comment == "solution found"
        ]
        solB = self.DengSolutionB_unfiltered[
            self.DengSolutionB_unfiltered.comment == "solution found"
        ]

        if hull_filter:
            solA = solA[solA.in_hull]
            solB = solB[solB.in_hull]

        solA_Cx = solA.groupby(["transNo"])["X"].mean()
        solA_Cy = solA.groupby(["transNo"])["Y"].mean()
        solA_Cz = solA.groupby(["transNo"])["Z"].mean()
        solB_Cx = solB.groupby(["transNo"])["X"].mean()
        solB_Cy = solB.groupby(["transNo"])["Y"].mean()
        solB_Cz = solB.groupby(["transNo"])["Z"].mean()

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.ephemeris.X_t.values, self.ephemeris.Y_t.values, self.ephemeris.Z_t.values, c="k")
        if len(solB) > 0:
            if beacon:
                ax.scatter(solA_Cx, solA_Cy, solA_Cz, c="green")
                ax.scatter(solB_Cx, solB_Cy, solB_Cz, c="cyan")
            else:
                ax.plot(solB_Cx, solB_Cy, solB_Cz, c="dimgray")

        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()


def positions_data_management(pos_type: str, inputWS: str, projectDB: str):
    """Aggregate Deng positioning CSV outputs into tblPositions_Deng."""
    files = [f for f in os.listdir(inputWS) if os.path.isfile(os.path.join(inputWS, f))]
    conn = sqlite3.connect(projectDB, timeout=30.0)
    if pos_type == "Deng":
        for f in files:
            solution = f[-5:-4]
            fishy = f[0:4]
            dat = pd.read_csv(os.path.join(inputWS, f))
            dat["solution"] = solution
            dat["Tag_ID"] = fishy
            dat.to_sql("tblPositions_Deng", con=conn, index=False, if_exists="append", chunksize=1000)
    conn.commit()
    conn.close()
