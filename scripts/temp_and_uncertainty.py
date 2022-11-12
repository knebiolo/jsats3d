# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:03:03 2021

@author: KNebiolo
"""
import os
import jsats3d
import pandas as pd
import sqlite3
import numpy as np
import time
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# identify workspace
posWS = r"J:\3870\013\Calcs\Paper\Output\coordinated_b\coordinated_b"
dbaseWS = r"J:\3870\013\Calcs\Paper\Data"
print ("Connected to directories")

# list files in directory
fish_files = os.listdir(posWS)

# loop over files and append to datase
pos_dat = pd.DataFrame()
for f in fish_files:
    dat = pd.read_csv(os.path.join(posWS,f))
    pos_dat = pos_dat.append(dat)
print ("Fish files read")
# now convert time to decimal seconds so we can interpolate
pos_dat['seconds'] = pd.DatetimeIndex(pos_dat.timestamp).astype(np.int64)/1.0e9


# extract temperature from project database    
temp_sql = "SELECT * from tblInterpolatedTemp"
dbName = 'cowlitz_2018_paper.db'
dbDir = os.path.join(dbaseWS,dbName)
conn = sqlite3.connect(dbDir,timeout = 30.0)
c = conn.cursor()
temp = pd.read_sql(temp_sql,con = conn)
print ("Temperature data extracted")
temp['seconds'] = pd.DatetimeIndex(temp.timeStamp).astype(np.int64)/1.0e9

# create a piecewise linear interpolator for temp
temp_fun = interp1d(temp.seconds.values,temp.C.values,bounds_error = False, fill_value = np.nan) 

# what is the temperature at every row in the pos_dat dataframe?
pos_dat['temp'] = temp_fun(pos_dat.seconds.values)
pos_dat.dropna(inplace = True)
#pos_dat = pos_dat[pos_dat.In_CH == True]


# plot and profit?
xfmt = mdates.DateFormatter('%m-%d')

# fig = plt.figure(figsize = (6,4))
# plt.xticks(rotation = 45)
# ax = fig.add_subplot(111)
# ax.scatter(pos_dat.temp.values,pos_dat['sigma z(m)'], label = r'sigma x(m)', color = 'b')
# ax.set_ylim(0,1000)
# plt.legend(loc = 2)
# #ax.set_ylabel('Celsius')
# plt.xticks(rotation = 45, fontsize = 7)
# #plt.title('Temperature as measured within Cowlitz Forebay')
# #plt.savefig(os.path.join(outputWS,'raw_temp.png'),bbox_inches = 'tight', dpi = 900)
# plt.show()


# fig = plt.figure(figsize = (6,4))
# plt.xticks(rotation = 45)
# ax = fig.add_subplot(111)
# ax.plot_date(pos_dat.timestamp.values,pos_dat['sigma x(m)'].values,linestyle = '-', label = r'X precision', color = 'red',ms = 0.1)
# #ax.plot_date(pos_dat.timestamp.values,pos_dat['sigma y(m)'].values,linestyle = '-', label = r'Y precision', color = 'blue',ms = 0.1)
# #ax.plot_date(pos_dat.timestamp.values,pos_dat['sigma z(m)'].values,linestyle = '-', label = r'Z precision', color = 'green',ms = 0.1)

# ax.xaxis.set_major_formatter(xfmt)
# ax.set_ylim(0,1)
# plt.legend(loc = 2)
# ax.set_ylabel('Precision (sigma)')
# plt.xticks(rotation = 45, fontsize = 7)
# #plt.title('Temperature as measured within Cowlitz Forebay')
# #plt.savefig(os.path.join(outputWS,'precision_over_time.png'),bbox_inches = 'tight', dpi = 900)
# plt.show()
