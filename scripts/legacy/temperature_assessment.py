'''Script Intent: Compare and contrast temperature sensors, see if we can develop
a function for C at t'''

# import modules
import pandas as pd
import numpy as np
import os
from datetime import date
import datetime as dt
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import statsmodels.api as sm
import matplotlib.dates as mdates
from scipy.interpolate import interp1d, UnivariateSpline
import sqlite3


# Declare workspace
outputWS = r"J:\3870\013\Calcs\Paper\Output"
inputWS = r"J:\3870\013\Calcs\Paper\Data"

# import data
temp = pd.read_csv(os.path.join(inputWS,"tblTemp.csv"))
temp['time_stamp'] = pd.to_datetime(temp.meas_dt)
temp['loc_dep_id'] = temp.location + "-" + temp.depth_ft.astype(np.str)
temp['temp_celcius'] = (temp.temp_f - 32) * 5./9.
temp['depth_m'] = np.round(temp.depth_ft / 3.28084, 2)

locs = temp.loc_dep_id.unique()
locDat = dict()
locSplineInterp = dict()
locLinearInterp = dict()
locEpochMin = dict()
locEpochMax = dict()
for i in locs:
    loc_dat = temp[temp.loc_dep_id ==i]
    loc_dat.sort_values('time_stamp', inplace = True)
    loc_dat['seconds'] = loc_dat.time_stamp.astype(np.int64)/1.0e9
    loc_dat.drop_duplicates('seconds',keep = 'first',inplace = True)
    loc_dat.set_index('seconds',inplace = True,drop = False)
    loc_dat.dropna(inplace = True)
    print ("Location %s data from %s through %s"%(i,loc_dat.time_stamp.min(),loc_dat.time_stamp.max()))
    locDat[i]= loc_dat
    #locSplineInterp[i] = interp1d(loc_dat.epoch_us.values,loc_dat.rolled_C.values,kind = 'cubic',bounds_error = False,fill_value = np.nan)
    locLinearInterp[i] = interp1d(loc_dat.seconds.values,loc_dat.temp_celcius.values,kind = 'linear',bounds_error = False,fill_value = np.nan)
    locEpochMin[i] = loc_dat.seconds.min()
    locEpochMax[i] = loc_dat.seconds.max()
    
# plot 3 temperature time series
xfmt = mdates.DateFormatter('%m-%d')



# get min and max epoch, make a range, interpolate values and plot against real values 
epochMin = min(locEpochMin.values())-0.100000
epochMax = max(locEpochMax.values())+0.100000
spline_temps = []
linear_temps = []
depths = []
epoch_range = np.linspace(epochMin,epochMax,10000)
ts = pd.to_datetime(epoch_range,unit = 's')
for i in epoch_range:
    temps = []
    depths = []
#    for j in locSplineInterp:
#        temps_s.append(locSplineInterp[j](i))
    for j in locLinearInterp:
        temps.append(locLinearInterp[j](i))
    #spline_temps.append(np.nanmean(temps_s))
    linear_temps.append(np.nanmean(temps))

        
#xfmt = mdates.DateFormatter('%m-%d')
#
## smoothed data with inerpolator
#fig = plt.figure(figsize = (12,4))
#plt.xticks(rotation = 45)
#ax1 = fig.add_subplot(111)
#ax1.plot_date(locDat['A'].timeStamp.values,locDat['A'].rolled_C.values, 'b-', label = r'A')
#ax1.plot_date(locDat['B'].timeStamp.values,locDat['B'].rolled_C.values, 'r-', label = r'B')
#ax1.plot_date(locDat['C'].timeStamp.values,locDat['C'].rolled_C.values, 'g-', label = r'C')
#ax1.plot_date(locDat['R09'].timeStamp.values,locDat['R09'].rolled_C.values, 'm-', label = r'R05')
#
##ax1.plot_date(ts,spline_temps, 'k-', label = r'cubic')
#ax1.plot_date(ts,linear_temps, 'm-', label = r'linear')
#ax1.xaxis.set_major_formatter(xfmt)
#
#plt.legend(loc = 2)
#ax1.set_ylabel('Celsius')
#plt.xticks(rotation = 45, fontsize = 8)
#plt.title('Smoothed and Interpolated Temperature, Cowlitz Forebay')
#plt.savefig(os.path.join(outputWS,'smoothed_temp_interpolator.png'),bbox_inches = 'tight')
#plt.show()

fig = plt.figure(figsize = (12,4))
plt.xticks(rotation = 45)
ax1 = fig.add_subplot(111)
#ax1.plot_date(locDat['A'].timeStamp.values,locDat['A'].C.values, 'k-', label = r'A',lw = 0.5)
#ax1.plot_date(locDat['B'].timeStamp.values,locDat['B'].C.values, 'k-', label = r'B',lw = 0.5)
#ax1.plot_date(locDat['C'].timeStamp.values,locDat['C'].C.values, 'k-', label = r'C',lw = 0.5)
#ax1.plot_date(locDat['R09'].timeStamp.values,locDat['R09'].C.values, 'k-', label = r'R09',lw = 0.5)

ax1.plot_date(ts,linear_temps, 'r-', label = r'linear',lw = 1)
ax1.xaxis.set_major_formatter(xfmt)

plt.legend(loc = 2)
ax1.set_ylabel('Celsius')
plt.xticks(rotation = 45, fontsize = 8)
plt.title('Raw Data with Interpolated Temperature, Cowlitz Forebay')
plt.savefig(os.path.join(outputWS,'raw_with_interpolated.png'),bbox_inches = 'tight')
plt.show()

# create pandas dataframe for interpolated temperature and write to a database
df_dict = {'timeStamp':ts,'C':linear_temps}
temp_df = pd.DataFrame.from_dict(df_dict,orient = 'columns')
temp_df.head()
conn = sqlite3.connect(os.path.join(inputWS,'cowlitz_2018_paper.db'), timeout = 30.0)
c = conn.cursor
temp_df.to_sql('tblInterpolatedTemp',con = conn, if_exists = 'replace')

fig = plt.figure(figsize = (6,4))
plt.xticks(rotation = 45)
ax = fig.add_subplot(111)
ax.plot_date(temp[(temp.location == 'A') & (temp.depth_m == np.round(1.64 / 3.28084, 2))].time_stamp.values,temp[(temp.location == 'A') & (temp.depth_m == np.round(1.64 / 3.28084, 2))].temp_celcius.values,linestyle = '-', label = r'%s m'%(np.round(1.64 / 3.28084, 2)), color = '#ADD8E6',ms = 0.1)
ax.plot_date(temp[(temp.location == 'A') & (temp.depth_m == np.round(4.92 / 3.28084, 2))].time_stamp.values,temp[(temp.location == 'A') & (temp.depth_m == np.round(4.92 / 3.28084, 2))].temp_celcius.values,linestyle = '-',label = r'%s m'%(np.round(4.92 / 3.28084, 2)), color = '#87CEFA',ms = 0.1)
ax.plot_date(temp[(temp.location == 'A') & (temp.depth_m == np.round(9.84 / 3.28084, 2))].time_stamp.values,temp[(temp.location == 'A') & (temp.depth_m == np.round(9.84 / 3.28084, 2))].temp_celcius.values,linestyle = '-',label = r'%s m'%(np.round(9.84 / 3.28084, 2)), color = '#6495ED',ms = 0.1)
ax.plot_date(temp[(temp.location == 'A') & (temp.depth_m == np.round(14.76 / 3.28084, 2))].time_stamp.values,temp[(temp.location == 'A') & (temp.depth_m == np.round(14.76 / 3.28084, 2))].temp_celcius.values,linestyle = '-',label = r'%s m'%(np.round(14.76 / 3.28084, 2)), color = '#4169E1',ms = 0.1)
ax.plot_date(temp[(temp.location == 'A') & (temp.depth_m == np.round(19.69 / 3.28084, 2))].time_stamp.values,temp[(temp.location == 'A') & (temp.depth_m == np.round(19.69 / 3.28084, 2))].temp_celcius.values,linestyle = '-',label = r'%s m'%(np.round(19.69 / 3.28084, 2)), color = '#0000FF',ms = 0.1)
ax.plot_date(temp[(temp.location == 'A') & (temp.depth_m == np.round(24.61 / 3.28084, 2))].time_stamp.values,temp[(temp.location == 'A') & (temp.depth_m == np.round(24.61 / 3.28084, 2))].temp_celcius.values,linestyle = '-',label = r'%s m'%(np.round(24.61 / 3.28084, 2)), color = '#0000CD',ms = 0.1)
ax.plot_date(temp[(temp.location == 'A') & (temp.depth_m == np.round(29.53 / 3.28084, 2))].time_stamp.values,temp[(temp.location == 'A') & (temp.depth_m == np.round(29.53 / 3.28084, 2))].temp_celcius.values,linestyle = '-',label = r'%s m'%(np.round(29.53 / 3.28084, 2)), color = '#00008B',ms = 0.1)
ax.plot_date(temp[(temp.location == 'A') & (temp.depth_m == np.round(39.37 / 3.28084, 2))].time_stamp.values,temp[(temp.location == 'A') & (temp.depth_m == np.round(39.37 / 3.28084, 2))].temp_celcius.values,linestyle = '-',label = r'%s m'%(np.round(39.37 / 3.28084, 2)), color = '#000080',ms = 0.1)
ax.plot_date(temp[(temp.location == 'A') & (temp.depth_m == np.round(47.57 / 3.28084, 2))].time_stamp.values,temp[(temp.location == 'A') & (temp.depth_m == np.round(47.57 / 3.28084, 2))].temp_celcius.values,linestyle = '-',label = r'%s m'%(np.round(47.57 / 3.28084, 2)), color = '#191970',ms = 0.1)
ax.plot_date(ts,linear_temps, 'r-', label = r'linear',lw = 1)
ax.xaxis.set_major_formatter(xfmt)
ax.set_ylim(5,25)
plt.legend(loc = 2)
ax.set_ylabel('Celsius')
plt.xticks(rotation = 45, fontsize = 7)
#plt.title('Temperature as measured within Cowlitz Forebay')
plt.savefig(os.path.join(outputWS,'raw_temp.png'),bbox_inches = 'tight', dpi = 900)
plt.show()
