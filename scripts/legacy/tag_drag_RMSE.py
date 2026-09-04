# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:21:04 2021

script intent:
    1) import algorithm solution and tag drag data
    2) create interpolator for X, Y and Z at t for algorithm solution and tag drag data
    3) create a time series and interpolate X Y Z for algorithm and tag drag at t
    4) calculate RMSE for each dimension and profit


@author: KNebiolo
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# identify WS
algWS = r"C:\Users\knebiolo\Desktop\jsats with DBSCAN\Output"
tagWS = r"J:\3870\013\Calcs\Old\Data"
outputWS = r"C:\Users\knebiolo\Desktop\jsats with DBSCAN\Output"
print ("connected to workspaces")

# import data
alg_dat = pd.read_csv(os.path.join(algWS,'21C2_solutionB.csv'))
alg_dat = alg_dat[alg_dat.comment == 'solution found']
tag_dat = pd.read_csv(os.path.join(tagWS,'tag_drag.csv'))
print ("imported raw data")

# transform the tag drag data - small numbers play nicer
dX = 568456.6072

dY = 5146298.133

tag_dat['X'] = tag_dat.Easting - dX
tag_dat['Y'] = tag_dat.Northing - dY

# do some data management 
tag_dat['timestamp'] = pd.to_datetime(tag_dat.GPS_Date + " " + tag_dat.GPS_Time) + pd.Timedelta(hours = 1)
index = pd.DatetimeIndex(tag_dat.timestamp.values)
tag_dat['seconds'] = index.astype(np.int64)//1.0e9
tag_dat['seconds'] = tag_dat.seconds.astype(np.float64)
tag_dat['Xbar'] = tag_dat.X.rolling(window = 8).mean()
tag_dat['Ybar'] = tag_dat.Y.rolling(window = 8).mean()
tag_dat['Zbar'] = tag_dat['POINT_Z'].rolling(window = 8).mean()
#tag_dat.dropna(inplace = True)

alg_dat['timestamp'] = pd.to_datetime(alg_dat['ToA'],unit='s')
alg_dat['Xbar'] = alg_dat.X.rolling(window = 8).mean()
alg_dat['Ybar'] = alg_dat.Y.rolling(window = 8).mean()
alg_dat['Zbar'] = alg_dat.Z.rolling(window = 8).mean()
alg_dat.dropna(inplace = True)

print ("completed data management")

# define time range - make everything congruent
min_time = alg_dat.ToA.min()
max_time = alg_dat.ToA.max()
tag_dat = tag_dat[(tag_dat.seconds >= min_time) & (tag_dat.seconds <= max_time)]

print ("congruent time ranges")

# fit a spline to each X, Y and Z
alg_x = interp1d(x = alg_dat.ToA.values, y = alg_dat.Xbar.values, kind = 'linear', bounds_error = False, fill_value = "extrapolate")
alg_y = interp1d(x = alg_dat.ToA.values, y = alg_dat.Ybar.values, kind = 'linear', bounds_error = False, fill_value = "extrapolate")
alg_z = interp1d(x = alg_dat.ToA.values, y = alg_dat.Zbar.values, kind = 'linear', bounds_error = False, fill_value = "extrapolate")

tag_x = interp1d(x = tag_dat.seconds.values, y = tag_dat.Xbar.values, kind = 'linear', bounds_error = False, fill_value = "extrapolate")
tag_y = interp1d(x = tag_dat.seconds.values, y = tag_dat.Ybar.values, kind = 'linear', bounds_error = False, fill_value = "extrapolate")
tag_z = interp1d(x = tag_dat.seconds.values, y = tag_dat.Zbar.values, kind = 'linear', bounds_error = False, fill_value = "extrapolate")
print ("fit linear splines")

# create a time series, get X, Y and Z
ts = np.arange(min_time+1,max_time-1,1)

alg_x_int = alg_x(ts)
alg_y_int = alg_y(ts)
alg_z_int = alg_z(ts)

alg_x_int = alg_x_int[np.logical_not(np.isnan(alg_x_int))]
alg_y_int = alg_y_int[np.logical_not(np.isnan(alg_y_int))]
alg_z_int = alg_z_int[np.logical_not(np.isnan(alg_z_int))]


tag_x_int = tag_x(ts)
tag_y_int = tag_y(ts)
tag_z_int = tag_z(ts)

tag_x_int = tag_x_int[np.logical_not(np.isnan(tag_x_int))]
tag_y_int = tag_y_int[np.logical_not(np.isnan(tag_y_int))]
tag_z_int = tag_z_int[np.logical_not(np.isnan(tag_z_int))]


# calculate RMSE for X Y and Z, plot and profit
MSE_x = mean_squared_error(tag_x_int,alg_x_int)
MSE_y = mean_squared_error(tag_y_int,alg_y_int)
MSE_z = mean_squared_error(tag_z_int,alg_z_int)

RMSE_x = np.sqrt(MSE_x)
RMSE_y = np.sqrt(MSE_y)
RMSE_z = np.sqrt(MSE_z)

print ("RMSE X: %s, Y: %s, Z: %s"%(RMSE_x,RMSE_y,RMSE_z))

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
#ax.plot(alg_x_int,alg_y_int,alg_z_int, c = 'magenta', label = 'algorithm')    # true data
#ax.plot(tag_x_int,tag_y_int,tag_z_int, c = 'cyan', label = 'tag drag')    # true data

ax.plot(alg_dat.Xbar.values,alg_dat.Ybar.values,alg_dat.Zbar.values, c = 'k', linestyle = '-', label = 'algorithm')    # true data
ax.plot(tag_dat.X.values,tag_dat.Y.values,tag_dat['POINT_Z'], c = 'k', linestyle = '--', label = 'tag drag')    # true data
ax.set_xlim(-50,50)
ax.set_ylim(-50,50)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
        
plt.show()  

fig = plt.figure()
ax = fig.add_subplot(111)#,projection = '3d')
#ax.plot(alg_x_int,alg_y_int,alg_z_int, c = 'magenta', label = 'algorithm')    # true data
#ax.plot(tag_x_int,tag_y_int,tag_z_int, c = 'cyan', label = 'tag drag')    # true data

ax.plot(alg_dat.Xbar.values,alg_dat.Ybar.values, c = 'k', linestyle = 'solid', label = 'algorithm')    # true data
ax.plot(tag_dat.X.values,tag_dat.Y.values, c = 'k', linestyle = 'dotted', label = 'tag drag')    # true data
ax.set_xlim(-50,50)
ax.set_ylim(-50,50)
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.set_zlabel('Z')
ax.legend()
plt.show()  