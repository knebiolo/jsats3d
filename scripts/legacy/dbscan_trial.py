# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 20:10:40 2022

@author: KNebiolo

Script Intent: implement dbscan on a clock fix data object - can it remove multipath?
"""
# import modules
from sklearn.cluster import DBSCAN
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# get data
dat = pd.read_csv(r"C:\Users\knebiolo\Desktop\jsats_Notebook_Test\Output\kats_test_dat.csv")
plt.scatter(dat.seconds,dat.DDoA,s = 3,c = 'red')


# find first and last time step
t_min = dat.seconds.min()
t_max = dat.seconds.max()

# create time series
ts = np.arange(t_min,t_max,37.)

# create linear interpolator
f = interp1d(dat.seconds,dat.DDoA,kind = 'linear')

# interpolate DDoA
ddoa = f(ts)

# plot interpolated data
plt.scatter(ts,ddoa,s = 2, c = 'blue')
plt.show()

# calculate euclidean distance between interpolated observations
euc_dist = np.sqrt(np.power(np.diff(ts),2)+np.power(np.diff(ddoa),2))
plt.hist(euc_dist)
ext_dist = pd.DataFrame(euc_dist).quantile(q = [0.90,0.95,0.99])
plt.show()

# apply dbscan and plot clusters
test = DBSCAN(eps = 37.0000127384 ,min_samples = 3, metric = 'euclidean').fit(np.vstack((ts,ddoa)).T)
plt.scatter(ts,ddoa,c = test.labels_,s = 2)
plt.show()

# create a dataframe of time, ddoa, and labels, identify the first new label 
results = pd.DataFrame.from_dict({'ts':ts,'ddoa':ddoa,'class':test.labels_})

# remove outliers and plot
filtered = results[results['class'] != -1]

plt.scatter(filtered.ts,filtered.ddoa, s= 2)
plt.show()