# -*- coding: utf-8 -*-
 # Module contains all of the objects and required for analysis of telemetry data

# import modules required for function dependencies
import numpy as np
import pandas as pd
import os
import sqlite3
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from matplotlib import rcParams
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import stats
import operator
import sklearn.mixture as mix
import sklearn.naive_bayes as nb
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity
import plotly.graph_objects as go
import skkda
#from sklearn.mixture import GMM
#from sklearn.naive_bayes.GaussianNB import GaussianNB
#from sklearn.naive_bayes import GaussianNB


font = {'family': 'serif','size': 8}
#print "Imported Modules"
rcParams['font.size'] = 8
rcParams['font.family'] = 'serif'

'''Software with the intent on creating standardized database and function for the 
import, cleaning and 3d positioning of acoustic telemetry data from J-SATs tagged fish.'''

def create_project_db(directory,dbName):
    '''function creates an empty, standardized sqlite databaes.  The end user can edit
    the database using a DB browser for sqlite found at: hhtp://sqlitebrowser.org/'''
    
    conn = sqlite3.connect(os.path.join(directory,dbName),timeout = 30.0)
    c = conn.cursor()
    # create a study parameter table 
    # create a tag table
    c.execute('''DROP TABLE IF EXISTS tblTag''')
    c.execute('''CREATE TABLE tblTag(ID INTEGER, hex_ID TEXT, Tag_ID TEXT,      
        releaseDate TIMESTAMP, pulseRate REAL, TagType TEXT, Species TEXT)''')
    # create a receiver table
    c.execute('''DROP TABLE IF EXISTS tblReceiver''')
    c.execute('''CREATE TABLE tblReceiver(Rec_ID TEXT, Type TEXT, Tag_ID TEXT, 
        Ref_Elev TEXT, X REAL, Y REAL, Z REAL, X_t REAL, Y_t REAL, Z_t REAL)''')                                             
    # create an unfiltered data table
#    c.execute('''DROP TABLE IF EXISTS tblDetectionRaw''')    
#    c.execute('''CREATE TABLE tblDetectionRaw(timeStamp TIMESTAMP, 
#        seconds REAL, Tag_ID TEXT, Rec_ID TEXT, FreqOff INTEGER, Amplitude INTEGER, 
#        NBW INTEGER, SNR REAL, Valid INTEGER, Pascals INTEGER, Celsius REAL)''')                                           
    # create a Water Surface Elevation Table for receivers referenced to the water surface
    c.execute('''DROP TABLE IF EXISTS tblWSEL''')
    c.execute('''CREATE TABLE tblWSEL(timeStamp TIMESTAMP, WSEL REAL)''')
    # create a Water Temperature Table 
    c.execute('''DROP TABLE IF EXISTS tblTemp''')
    c.execute('''CREATE TABLE tblTemp(timeStamp TIMESTAMP, C REAL, location TEXT)''')                 

def set_study_parameters(utc_conv,bm_elev,bm_elev_units,output_units,masterReceiver,synch_time,dbName):
    '''Function sets parameters for predictor variables used in the naive bayes
    classifier
    
    det = number of detections to look forward and backward in times for detection
    history strings
    
    duration = moving window around each detection, used to calculate the noise 
    ratio and number of fish present (fish count)
    
    '''
    conn = sqlite3.connect(dbName, timeout=30.0)
    c = conn.cursor()
    params = [(utc_conv,bm_elev,bm_elev_units,output_units,masterReceiver,synch_time)]
    c.execute('''DROP TABLE IF EXISTS tblStudyParameters''')
    c.execute('''CREATE TABLE tblStudyParameters(UTC_Conv INTEGER,BM_Elev REAL, BM_Elev_Units TEXT, Output_Units TEXT, masterReceiver TEXT, synch_time TIMESTAMP)''')

    conn.executemany('INSERT INTO tblStudyParameters VALUES (?,?,?,?,?,?)',params)
    conn.commit()
    c.close()         
                        
def study_data_import(dataFrame,dbName,tblName):
    '''function imports formatted data into project database. The code in its current 
    function does not check for inconsistencies with data structures.  If you're 
    shit isn't right, this isn't going to work for you.  Make sure your table data 
    structures match exactly, that column names and datatypes match.  
        
    dataFrame = pandas dataframe imported from your structured file.  
    dbName = full directory path to project database
    tblName = the name of the data you can import to,meant for tblMasterTag and 
    tblMasterReceiver'''
    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    dataFrame.to_sql(tblName,con = conn,index = False, if_exists = 'append') 
    conn.commit()   
    c.close()
    
def temp_interpolator(projectDB,interp_type):
    '''Python function that creates a temperature interpolator
    interp_type describes the interpolator type, either cubic or linear
    
    At project initiation, we create the interpolator and pickle the object in the 
    input data directory'''
    
    conn = sqlite3.connect(projectDB, timeout=30.0)
    c = conn.cursor()
    temp = pd.read_sql('SELECT * FROM tblInterpolatedTemp',con = conn)
    c.close() 
    temp['timeStamp'] = pd.to_datetime(temp.timeStamp)
    temp.sort_values('timeStamp', inplace = True)
    seconds = pd.DatetimeIndex(temp.timeStamp).astype(np.int64)/1.0e9
    temp['seconds'] = seconds.values
    temp.drop_duplicates('seconds',keep = 'first',inplace = True)
    temp.set_index('seconds',inplace = True,drop = False)
    interpolator = dict()
    interpolator['a'] = interp1d(temp.seconds.values,temp.C.values,kind = interp_type,bounds_error = False,fill_value = np.nan)
    
#    temp = pd.read_sql('SELECT * FROM tblTemp',con = conn)
#    c.close() 
#    temp['timeStamp'] = pd.to_datetime(temp.timeStamp)
#    locs = temp.location.unique()
#    interpolator = dict()
#    for i in locs:
#        loc_dat = temp[temp.location ==i]
#        loc_dat.sort_values('timeStamp', inplace = True)
#        epoch = pd.DatetimeIndex(loc_dat.timeStamp).astype(np.int64)/1000
#        loc_dat['epoch_us'] = epoch.values
#        loc_dat.drop_duplicates('epoch_us',keep = 'first',inplace = True)
#        loc_dat.set_index('epoch_us',inplace = True,drop = False)
#        if i != 'R09':
#            loc_dat['rolled_C'] = loc_dat['C'].rolling(120).mean()
#        else:
#            loc_dat['rolled_C'] = loc_dat['C']
#        loc_dat.dropna(inplace = True)
#        interpolator[i] = interp1d(loc_dat.epoch_us.values,loc_dat.rolled_C.values,kind = interp_type,bounds_error = False,fill_value = np.nan)
    return interpolator

def avg_temp(row,temp_interpolator):
    seconds = row[1]['seconds']
    temps = []
    for i in temp_interpolator:
        temps.append(temp_interpolator[i](seconds))
    return np.nanmean(temps)

def teknologic_import(UTC_conv,inputWS, dbName, recName):
    '''function imports raw data from Teknologic cabled receivers''' 
    def timeStamp (row):                                                        # function that concatenates the teknologic time string
        year = row['Year']
        month = row['Month']
        day = row['Day']
        hours = row['Hour']
        minutes = row['Minute']
        seconds = row['Second']
        return datetime(year,month,day,hours,minutes,seconds)
    conn = sqlite3.connect(dbName)
    synch_time = pd.read_sql('SElECT synch_time from tblStudyParameters', con = conn).synch_time.values[0]
    synch_time = pd.to_datetime(synch_time)
    c = conn.cursor()
    files = os.listdir(inputWS)
    det_data = pd.DataFrame(columns = ['Sequence','Year', 'Month', 'Day', 'Hour', 'Minute','Second','UnixSeconds','Microseconds','timeStamp','seconds','Tag_ID','Rec_ID','FreqOff','Amplitude','NBW','SNR','Valid', 'Pascals','Celsius'])
    for f in files:
        detFile = os.path.join(inputWS,f)
        det = pd.read_csv(detFile, header = None, names = ['Sequence','Year', 'Month', 'Day', 'Hour', 'Minute','Second','UnixSeconds','Microseconds','Tag_ID','FreqOff','Amplitude','NBW','SNR','Valid', 'Pascals','Celsius'])
        det['Rec_ID'] = np.repeat(recName, len(det))
        det['Tag_ID'] = det['Tag_ID'].str.strip()
        det['timeStamp'] = det.apply(timeStamp,axis = 1)                           # apply timestamp to every row in the dataframe  
        index = pd.DatetimeIndex(det.timeStamp.values)
        det['seconds'] = index.astype(np.int64)//1.0e9
        det['seconds'] = det.seconds.astype(np.float64)
        det['seconds'] = np.round(det.seconds + (det.Microseconds/1.0e6),6)
        '''for synch 1 only, there was a power outage and clocks went sideways - we can't use data from these times:'''
        det1 = det[(det.timeStamp <= '2019-05-15 10:00')]
        det2 = det[(det.timeStamp >= '2019-05-15 14:30')]
        det = det1.append(det2)
#        '''for synch 2 only, need to transform timestamps for these receivers comment out on other runs''' 
#        if recName == 'R01' or recName == 'R02' or recName == 'R03' or recName == 'R05' or recName == 'R08' or recName == 'R09':
#            det['timeStamp'] = det.timeStamp + pd.Timedelta(1,'h')   
#        '''for synch 4,6 need to transform timestamps for receivers R09''' 
#        if recName == 'R09':
#            det['timeStamp'] = det.timeStamp - pd.Timedelta(1,'h')   


        
#        ''' for synch 5, there is some funky clocks between 8/16/19 11:18 and 8/16/19 11:22 - going to filter out those times'''
#        det = det[(det.timeStamp <= pd.to_datetime('2019-08-16 11:18')) | (det.timeStamp >= pd.to_datetime('2019-08-16 11:22'))]
        
        ''' for synch 6, we run out of temperature - remove hits before then'''
        det = det[(det.timeStamp <= pd.to_datetime('2019-09-05 12:36:55.401469'))]
        
        print ("Length of dataframe before removing detections before synchronization time is %s"%(len(det)))
        det = det[det.timeStamp > synch_time]
        print ("Length of dataframe after removing detections before synchronization time is %s"%(len(det)))
        det.sort_values(by = 'seconds', ascending = True, inplace = True)
        det_data = det_data.append(det)
    det_data.drop(['Sequence','Year','Month','Day','Hour','Minute','Second','UnixSeconds','Microseconds'],axis = 1, inplace = True)
    det_data.drop_duplicates(keep = 'first', inplace = True)                      # wtf duplicate data
    det_data.to_sql('tblDetectionRaw',conn,'sqlite',if_exists = 'append', index = False)                               # write dataframe to full radio table in the sqlite database
    conn.commit()   
    c.close()
    #del det
    # create and apply timestamp

def acoustic_data_import(site,recType,rawDataFiles,projectDB):
    '''Function allows import from Teknologic receivers.  User must prompt with 
    the site (Rec_ID), receiver Type (either Teknologic, ...), the directory that 
    contains the raw data files, and the directory of the project database''' 
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    UTC_conv = pd.read_sql('SELECT UTC_Conv FROM tblStudyParameters',con = conn).UTC_Conv.values[0]
    c.close()
    if recType == 'Teknologic':
        teknologic_import(UTC_conv,rawDataFiles,projectDB,site)

class beacon_epoch():
    '''Python class object to hold data for and methods to find the transmission epoch 
    for beacon tags.
    
    We need to first enumerate beacon tag transmissions at each host receiver by 
    looking at lags between each detection. If the lag is less than 1/2 
    a pulse rate, we are still within the same beacon pulse epoch.  However, if 
    the lag is greater than 1/2 of the pulse rate, it is in the next epoch.  
    
    Then we will iterate over each beacon tag transmission and apply a moving window of:
    (0.5 rate < ToA < 0.5 rate) around beacon tag transmissions at all other receivers.
    If I detection is heard within this window at a neighboring receiver, then we 
    apply the corresponding epoch number.  If it is out of the window, it is in 
    another epoch.
    
    If the clocks drift too much, this will not work.'''
    def __init__(self,tag,projectDB,scratchWS):
        self.tag = tag
        conn = sqlite3.connect(projectDB, timeout = 30.0)
        self.projectDB = projectDB
        
        # get this tag's pulse rate from the project database
        tagSQL = "SELECT pulseRate, TagType from tblTag WHERE Tag_ID = '%s'"%(tag)
        tagDat = pd.read_sql(tagSQL, con = conn)
        self.pulseRate = tagDat.pulseRate.values[0]
        self.tagType = tagDat.TagType.values[0]
        receiver = pd.read_sql("SELECT Rec_ID from tblReceiver WHERE Tag_ID = '%s'"%(tag),con = conn).Rec_ID.values[0]
        
        # get data for this tag at the host receiver and all other receivers
        host_dat_SQL = "SELECT * from tblDetectionRaw WHERE Tag_ID = '%s' AND Rec_ID = '%s'"%(tag,receiver)
        self.host_dat = pd.read_sql(host_dat_SQL, con = conn)
        child_dat_SQL = "SELECT * from tblDetectionRaw WHERE Tag_ID = '%s' AND Rec_ID != '%s'"%(tag,receiver)
        self.child_dat = pd.read_sql(child_dat_SQL, con = conn)
        # create a multilevel index on Rec_ID, Tag_ID and epoch_us in the host data frame and child dataframe
        i_arrays = [self.host_dat.Rec_ID.values,self.host_dat.Tag_ID.values,self.host_dat.seconds.values]
        i_tuples = list(zip(*i_arrays)) 
        i_index = pd.MultiIndex.from_tuples(i_tuples,names = ['Rec_ID','Tag_ID','seconds'])
        self.host_dat.set_index(i_index, inplace = True)
        self.host_dat.sort_index(level = 'seconds', inplace = True)
         
        
        j_arrays = [self.child_dat.Rec_ID.values,self.child_dat.Tag_ID.values,self.child_dat.seconds.values]
        j_tuples = list(zip(*j_arrays))
        j_index = pd.MultiIndex.from_tuples(j_tuples,names = ['Rec_ID','Tag_ID','seconds'])
        self.child_dat.set_index(j_index, inplace = True)
        self.child_dat.sort_index(level = 'seconds', inplace = True)  
         
        # people are sloppy, clean up the field crew's mess
        self.host_dat.drop_duplicates(keep = 'first', inplace = True) 
        self.child_dat.drop_duplicates(keep = 'first', inplace = True) 
       
        # create a cratch workspace
        self.scratchWS = scratchWS
     
    def host_receiver_enumeration(self):          
        # Next calculate the transmission number
        self.host_dat['lag'] = self.host_dat.seconds.diff()             # let's first calculate the lag between detections.
        self.host_dat.fillna(0,inplace = True)
        self.host_dat['metronome_transmission'] = np.zeros(len(self.host_dat)) # create an empty column for transmission number
        transNo = 0
        for i in self.host_dat.iterrows():                                     # for every row in our dataframe 
            curr_lag = i[1]['lag']                                             # get the current lag
            if curr_lag < 0.5 * self.pulseRate:                                # if the current lag is less than 1/2 the transmission rate, we are in the same epoch
                self.host_dat.at[i[0],'metronome_transmission'] = transNo      # set value
            else:                                                              # if it isn't, we are in the next epoch
                transNo = transNo + 1                                          # increase transmission number enumerator by 1
                self.host_dat.at[i[0],'metronome_transmission'] = transNo      # set value        
        #self.host_dat[['Tag_ID','Rec_ID','timeStamp','metronome_transmission']].to_csv(os.path.join(self.scratchWS,'check.csv'),index = False)
        conn = sqlite3.connect(self.projectDB, timeout = 30.0)
        c = conn.cursor()
        #self.host_dat.drop(labels  = 'lag',axis = 1, inplace = True)
        self.host_dat.to_sql('tblMetronomeUnfiltered',conn, 'sqlite', if_exists = 'append', index = False)  
        c.close()
        
    def adjacent_receiver_enumeration(self):
        receivers = self.child_dat.Rec_ID.unique()                             # get a list of receivers from the child data frame
        for i in receivers:                                                    # for each reciever
            rec_dat = self.child_dat[self.child_dat.Rec_ID == i]               # get this reciever's data
            rec_dat['metronome_transmission'] = np.repeat(np.nan,len(rec_dat))         # create a metronome transmission column
            counter = 0
            for j in self.host_dat.metronome_transmission.values:              # for every unique transmission in the host data (eww)
                trans_time = self.host_dat[self.host_dat.metronome_transmission == j].seconds.min() # get the time of transmission of this beacon tag pulse - we can have more than 1 b/c of multipath so take the min time 
                print ("Host transmisstion number %s sent at %s"%(j,trans_time))
                rows = rec_dat[(rec_dat.seconds > (trans_time - (0.5*self.pulseRate))) & 
                               (rec_dat.seconds < (trans_time + (0.5*self.pulseRate)))] # restrict the child dataframe to those detections within +/- 1/2 a pulse rate from the current time of transmission                    
                if len(rows) > 0:                                              # if the length of the returned rows is greater than 1, this beacon tag transmission was heard at the adjacent receiver
                    for k in rows.iterrows():                                  # for every returned row - we could have multipath error.. . .
                        print ("Beacon tag transmission number %s heard at receiver %s at us %s"%(j,i,round(k[0][2],6)))
                        rec_dat.at[k[0],'metronome_transmission'] = j   # write the metronome transmission sequence to the adjancent receiver data
                counter = counter + 1
            conn = sqlite3.connect(self.projectDB, timeout = 30.0)
            rec_dat.to_sql('tblMetronomeUnfiltered',conn, 'sqlite', if_exists = 'append', index = False)  
        
        # create an index on tblMetronomeUnfiltered         
        conn = sqlite3.connect(self.projectDB, timeout = 30.0)
        rec_dat.to_sql('tblMetronomeUnfiltered',conn, 'sqlite', if_exists = 'append', index = False)  
        c = conn.cursor()        
        conn.commit()
        c.close()
    
    def indexer(self):
        conn = sqlite3.connect(self.projectDB, timeout = 30.0)
        c = conn.cursor()        
        c.execute('''CREATE INDEX idx_combined_metronome_unfiltered ON tblMetronomeUnfiltered (Rec_ID, Tag_ID, seconds)''')
        conn.commit()
        c.close()

class multipath_data_object():
    '''multipath data object class.'''
    def __init__(self,tag,projectDB,scratchWS,metronome = False):
        self.tag = tag
        self.metronome = metronome
        self.projectDB = projectDB
        conn = sqlite3.connect(projectDB, timeout = 30.0)
        c = conn.cursor()
        # get this tag's pulse rate from the project database
        tagSQL = "SELECT pulseRate, TagType from tblTag WHERE Tag_ID = '%s'"%(tag)
        tagDat = pd.read_sql(tagSQL, con = conn)
        self.pulseRate = tagDat.pulseRate.values[0]
        self.tagType = tagDat.TagType.values[0]
        self.master_receiver = pd.read_sql('SELECT masterReceiver from tblStudyParameters',con = conn).masterReceiver.values[0]
        # get data for this tag across all receivers
        if metronome == True:
            datSQL = "SELECT * from tblMetronomeUnfiltered WHERE Tag_ID = '%s'"%(tag)
        else:
            datSQL = "SELECT * from tblDetectionClockFixed WHERE Tag_ID = '%s'"%(tag)
        self.data = pd.read_sql(datSQL, con = conn)
        
        self.scratchWS = scratchWS
        
        # this fish may have not been recorded, if it doesn't exist yet, set it to empty, if it does, create multilevel index
        if len(self.data) > 0:
            self.empty = False
            # create a multilevel index on Rec_ID, Tag_ID and epoch_us
            if metronome == False:
                i_arrays = [self.data.Rec_ID.values,self.data.Tag_ID.values,self.data.seconds_fix.values]
                i_tuples = list(zip(*i_arrays))
                index = pd.MultiIndex.from_tuples(i_tuples,names = ['Rec_ID','Tag_ID','seconds_fix'])
                self.data.set_index(index, inplace = True)
            else:
                i_arrays = [self.data.Rec_ID.values,self.data.Tag_ID.values,self.data.seconds.values]
                i_tuples = list(zip(*i_arrays))
                index = pd.MultiIndex.from_tuples(i_tuples,names = ['Rec_ID','Tag_ID','seconds'])
                self.data.set_index(index, inplace = True)
            # calculate the transmission number 
            '''
            We have ambiguity here, we have to hope and pray that the clocks 
            were initially synched.  Not cool man, not cool.
            '''
            if metronome == False:                                             # we have already enumerated transmission numbers during the metronome procedure          
                if self.tagType == 'study':
                    firstDet = self.data.seconds_fix.min()                        # find the first ever detection of this tag
                    epochDiff = self.data.seconds_fix.values - firstDet           # calculate the difference between the time and the first detection
                    #transNo = epochDiff//(self.pulseRate * 1e6)               # calculate the transmission number
                    transNo = np.round(epochDiff/self.pulseRate,0) # calculate the transmission number
                    self.data['transNo'] = transNo                             # write it to the dataframe    
                else:                   
                    ''' note this will fail if the clocks have drifted so much 
                    we are in a new epoch - there's no way for us to tell'''
                    self.data['transNo'] = np.repeat(np.nan,len(self.data))    # create an empty column for transmission number
                    transNo = 0
                    # if we are removing multipath at this phase in the analysis for a receiver, then that means we don't know it's position, meaning it's clock hasn't been fixed yet.  So we need to define the host receiver here, the receiver with the min and max timestamp
                    det_counts = self.data.groupby(['Rec_ID'])['seconds_fix'].count().to_frame().rename(columns = {'seconds_fix':'row_count'}) # get number of rows by receiver ID
                    det_counts.reset_index(drop = False, inplace = True)       # reset the index
                    max_count = det_counts.row_count.max()                     # get the max row count
                    host_rec = det_counts[det_counts.row_count == max_count].Rec_ID.values[0] # the host receiver is the one with the maximum number of rows 
                    host_dat = self.data[self.data.Rec_ID == host_rec]         # extract the host data
                    host_dat['lag'] = host_dat.seconds_fix.diff()              # let's first calculate the lag between detections.
                    host_dat.fillna(0,inplace = True)
                    for i in host_dat.iterrows():                              # for every row in our dataframe 
                        curr_lag = i[1]['lag']                                 # get the current lag
                        if curr_lag < 0.5 * self.pulseRate:                    # if the current lag is less than 1/2 the transmission rate, we are in the same epoch
                            host_dat.at[i[0],'transNo'] = transNo              # set value
                            self.data.at[i[0],'transNo'] = transNo             # set value                            
                        else:                                                  # if it isn't, we are in the next epoch
                            transNo = transNo + 1                              # increase transmission number enumerator by 1
                            host_dat.at[i[0],'transNo'] = transNo              # set value        
                            self.data.at[i[0],'transNo'] = transNo             # set value
                    
                    rec_dat = self.data[self.data.Rec_ID != host_rec]          # get every other reciever's data                        
                    for i in host_dat.transNo.values:                          # for every unique transmission in the host data (eww)
                        trans_time = host_dat[host_dat.transNo == i].seconds_fix.min() # get the time of transmission of this beacon tag pulse - we can have more than 1 b/c of multipath so take the min time 
                        rows = rec_dat[(rec_dat.seconds_fix > (trans_time - (0.5*self.pulseRate))) & 
                                       (rec_dat.seconds_fix < (trans_time + (0.5*self.pulseRate)))] # restrict the child dataframe to those detections within +/- 1/2 a pulse rate from the current time of transmission                    
                        if len(rows) > 0:                                              # if the length of the returned rows is greater than 1, this beacon tag transmission was heard at the adjacent receiver
                            for k in rows.iterrows():                                  # for every returned row - we could have multipath error.. . .
                                self.data.at[k[0],'transNo'] = i               # write the metronome transmission sequence to the adjancent receiver data

            
            # now we need to get rid of duplicates, because let's be honest, we're sloppy.
            self.data.drop_duplicates(keep = 'first', inplace = True) # let's keep the first detection
        else:
            self.empty = True
        c.close()
        
def multipath_2(multipath_object):
    '''A hopefully more efficient multipath, rather than iterating over rows...
    twice... this function uses a pandas groupby transmission number and rank by
    epoch.  Rank 0 = 0, Rank > 0 = 1
    '''
    receivers = np.unique(multipath_object.data.Rec_ID.values)                 # identify unique receivers to iterate through
    if multipath_object.empty == False:
        for i in receivers:                                                    # for each receiver
            recDat = multipath_object.data[multipath_object.data.Rec_ID == i]  # extract data for receiver i
            if multipath_object.metronome == False:
                recDat.sort_index(level = 'seconds_fix', inplace = True)
                # group by transmission number and rank the epoch, then rename series det rank for detection rank
                grouped = recDat.groupby(by = ['transNo'])['seconds_fix'].rank().rename('det_rank')
                recDat = recDat.join(grouped, how = 'left')                        # join i
                # create a multilevel index on transmission number and epoch
                transNo = recDat.transNo.values
                seconds_fix = recDat.seconds_fix.values
                i_arrays = [transNo,seconds_fix]
                i_tuples = list(zip(*i_arrays))
                index = pd.MultiIndex.from_tuples(i_tuples,names = ['transNo','seconds_fix'])
                recDat.set_index(index, inplace = True)
            else:
                recDat.sort_index(level = 'seconds', inplace = True)
                print (recDat.head(20))
                # group by transmission number and rank the epoch, then rename series det rank for detection rank
                grouped = recDat.groupby(by = ['metronome_transmission'])['seconds'].rank().rename('det_rank')
                recDat = recDat.join(grouped, how = 'left')                        # join it   
                # create a multilevel index on transmission number and epoch
                recDat.rename(columns = {'metronome_transmission':'transNo'},inplace = True)
                recDat.drop('lag',axis = 1, inplace = True)
                transNo = recDat.transNo.values
                seconds = recDat.seconds.values
                i_arrays = [transNo,seconds]
                i_tuples = list(zip(*i_arrays))
                index = pd.MultiIndex.from_tuples(i_tuples,names = ['transNo','seconds'])
                recDat.set_index(index, inplace = True)  
            
            
            
            recDat.reset_index(drop = True, inplace = True)                    # clean up the dataframe
            #del grouped, transNo, seconds, i_arrays, i_tuples, index          # leave no trace
            # create function for multipath classification and apply to dataframe
            def multi_apply(row):
                if row['det_rank'] > 1.0:
                    return 1
                else:
                    return 0                    
            recDat['multipath'] = recDat.apply(multi_apply,axis = 1)  
            
            primary = recDat[recDat.multipath == 0]
            mutli = recDat[recDat.multipath == 1]
            # 3d plot
#            fig = plt.figure(figsize = (6,6))
#            ax = fig.add_subplot(111,projection = '3d')
#            ax.plot(primary.Amplitude.values, primary.SNR.values, primary.NBW.values,'ko', label = 'primary')
#            ax.plot(mutli.Amplitude.values, mutli.SNR.values, mutli.NBW.values,'ro', alpha = 0.2,label = 'multipath')
#            ax.legend()
#            ax.set_xlabel('Amplitude')
#            ax.set_ylabel('Signal to Noise Ratio')
#            ax.set_zlabel('Noise in Bandwidth')
#            #plt.savefig(os.path.join(figureWS,"filter_variables.png"),bbox_inches = 'tight')
#            plt.show()

            if multipath_object.metronome == False:
                recDat.to_csv(os.path.join(multipath_object.scratchWS,'%s_%s_multipath.csv'%(multipath_object.tag,i)),index = False, float_format='%.6f')
            else:
                conn = sqlite3.connect(multipath_object.projectDB, timeout = 30.0)
                c = conn.cursor()
                recDat.to_sql('tblMetronomeFiltered',conn, 'sqlite', if_exists = 'append', index = False)
                c.close()
        if multipath_object.metronome == True:
            conn = sqlite3.connect(multipath_object.projectDB, timeout = 30.0)
            c = conn.cursor()
            # create an index on tblMetronomeUnfiltered 
            c.execute('''CREATE INDEX idx_combined_metronome_filtered ON tblMetronomeFiltered (Rec_ID, Tag_ID, seconds)''')
            conn.commit()
            c.close()
         
def multipath_data_management(inputWS,projectDB,primary = True):
    # As soon as I figure out how to do this function is moot.
    if primary == True:
        tblName = 'tblDetectionFilterPrimary'
    else:
        tblName = 'tblDetectionFilterSecondary'
    files = os.listdir(inputWS)
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    for f in files:
        dat = pd.read_csv(os.path.join(inputWS,f))
        #dat.drop(['FreqOff','Valid'],axis = 1,inplace = True)
        dat.to_sql(tblName,con = conn,index = False, if_exists = 'append', chunksize = 1000)
        os.remove(os.path.join(inputWS,f))
        del dat  
    # create an index on tblMetronomeUnfiltered 
    if primary == True:
        c.execute('''CREATE INDEX idx_combined_primary_filter ON tblDetectionFilterPrimary (Rec_ID, Tag_ID, seconds_fix, transNo)''')
    else:
        c.execute('''CREATE INDEX idx_combined_second_filter ON tblDetectionFilterSecondary (Rec_ID, Tag_ID, seconds_fix, transNo)''')

    conn.commit()              
    c.close()               

def mulitpath_classifier(tag,projectDB,outputWS,beacon = False):
    # get data 
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    dat = pd.read_sql('SELECT * FROM tblDetectionFilterPrimary WHERE Tag_ID == "%s"'%(tag),con = conn)
    if beacon == True:
        rec_ID = pd.read_sql('SELECT Rec_ID FROM tblReceiver WHERE Tag_ID == "%s"'%(tag),con = conn).Rec_ID.values[0]
    c.close()
    dat = dat[dat.SNR > 0.0]
    if len(dat) > 0:
        recs = dat.Rec_ID.unique()
        for i in recs:
            rec_dat = dat[dat.Rec_ID == i]
            # set indices
            rec_dat.set_index('transNo',inplace = True,drop = False)
            rec_dat.sort_index(inplace = True)
            # make sure epoch is an integer
            rec_dat['logAmp'] = np.log10(rec_dat.Amplitude.values)
            rec_dat['logNBW'] = np.log10(rec_dat.NBW.values)
            rec_dat['logSNR'] = np.log10(rec_dat.SNR.values)
            print (rec_dat.head())
            # let's make a classifier, if we have training data make a supervised, if not, make an unsupervised 
            # to make a supervised classifier, we need training data       
       
            if beacon == False:
                train_dat = rec_dat[(rec_dat.SNR > 0.0)]
            else:
                train_dat = rec_dat[(rec_dat.Rec_ID != rec_ID) & (rec_dat.SNR > 0.0)]
            train_idx = train_dat[train_dat.det_rank > 1.0].index.tolist()     # training data occurs when we have a primary detection and known multipath - aka, any detection with a detection rank greater than 1
            train_dat = train_dat.loc[train_idx]                               # get the indices for these rows 
            #train_dat = []
            if len(train_dat) > 3: 
                print (train_dat.groupby(['multipath'])[['logAmp','logNBW','logSNR']].describe())

                X = train_dat[['logSNR','logAmp','logNBW']]                       # get predictors
                y = train_dat[['multipath']]                                   # get response variable
                classifier = nb.GaussianNB()                                      # create classifier object
                classifier.fit(X,y)                                            # train a classifier        
                
                # make a prediction on the multipath status of the detection 
                rec_dat['multipath_prediction'] = classifier.predict(rec_dat[['logSNR','logAmp','logNBW']]) # predict classes of unknown detections            
                
            else: # if we don't have any training data, perform a supervised classifier
                # a gaussian mixture model is like a k-means except it is statistical 
                if len(rec_dat) > 50:
                    gmm = mix.GaussianMixture(n_components = 2).fit(rec_dat[['logSNR','logAmp','logNBW']])
                    probs = gmm.predict_proba(rec_dat[['logSNR','logAmp','logNBW']])    
                    # classify each row - decision model is simple, whichever has the larger probability 
                    classArr = []
                    for j in probs:
                        if j[0] > j[1]:
                            classArr.append('A')
                        else:
                            classArr.append('B')
                    rec_dat['DetClass'] = classArr
                           
                    # ok, so we have two detection classes, which one is better?  The one with the larger SNR
                    A = rec_dat[rec_dat.DetClass == 'A']
                    A_mean = np.median(A.SNR.values)
                    B = rec_dat[rec_dat.DetClass == 'B']
                    B_mean = np.median(B.SNR.values)
                    rec_dat.reset_index(drop = True,inplace = True)
                    rec_dat.set_index('DetClass',drop = False,inplace = True)
                    if A_mean > B_mean: # the detection class with a larger SNR is the obvious choice
                        rec_dat.loc['A','multipath_prediction'] = 0
                        rec_dat.loc['B','multipath_prediction'] = 1                    
                    else:
                        rec_dat.loc['A','multipath_prediction'] = 1
                        rec_dat.loc['B','multipath_prediction'] = 0             
           
    
                    # perform some data management 
                    rec_dat.reset_index(drop = True,inplace = True)
                    rec_dat.drop('DetClass',axis = 1,inplace = True)
                else:
                    rec_dat['multipath_prediction'] = np.repeat(0,len(rec_dat))

                #rec_dat['multipath_prediction'] = np.zeros(len(rec_dat))
            if beacon == True:
               host_idx = rec_dat[rec_dat.Rec_ID == rec_ID].index
               rec_dat.at[host_idx,'multipath_prediction'] = 0
            # export data 
            rec_dat.to_csv(os.path.join(outputWS,'multipath_predict_%s_at_%s.csv'%(tag,i)), index = False, float_format='%.6f')

def sos_apply(row):
    # fix the time of arrival at receiver j
    # create a function to calculate speed of sound with temperature (C) as predictor and plot 
    # See: https://en.wikipedia.org/wiki/Speed_of_sound for data
    temp = row['Celsius']
    arr_temp = np.arange(4,30.5,0.5)                           # degrees C
    sps = np.array([1421.62,1423.9,1426.15,1428.38,1430.58,1432.75,1434.9,1437.02,
                    1439.12,1441.19,1443.23,1445.25,1447.25,1449.22,1451.17,1453.09,
                    1454.99,1456.87,1458.72,1460.55,1462.36,1464.14,1465.91,1467.65,
                    1469.36,1471.06,1472.73,1474.38,1476.01,1477.62,1479.21,1480.77,
                    1482.32,1483.84,1485.35,1486.83,1488.29,1489.74,1491.16,1492.56,
                    1493.95,1495.32,1496.66,1497.99,1499.3,1500.59,1501.86,1503.11,
                    1504.35,1505.56,1506.76,1507.94,1509.1])   # m/s
    sps = sps * 3.2804                                                             # conversion to ft/s
    f = interp1d(arr_temp,sps,kind = 'cubic')
    return f(temp)

def sos(temp):
    # fix the time of arrival at receiver j
    # create a function to calculate speed of sound with temperature (C) as predictor and plot 
    # See: https://en.wikipedia.org/wiki/Speed_of_sound for data
    arr_temp = np.arange(4,30.5,0.5)                           # degrees C
    sps = np.array([1421.62,1423.9,1426.15,1428.38,1430.58,1432.75,1434.9,1437.02,
                    1439.12,1441.19,1443.23,1445.25,1447.25,1449.22,1451.17,1453.09,
                    1454.99,1456.87,1458.72,1460.55,1462.36,1464.14,1465.91,1467.65,
                    1469.36,1471.06,1472.73,1474.38,1476.01,1477.62,1479.21,1480.77,
                    1482.32,1483.84,1485.35,1486.83,1488.29,1489.74,1491.16,1492.56,
                    1493.95,1495.32,1496.66,1497.99,1499.3,1500.59,1501.86,1503.11,
                    1504.35,1505.56,1506.76,1507.94,1509.1])   # m/s    sps = sps * 3.2804                                                             # conversion to ft/s
    f = interp1d(arr_temp,sps,kind = 'cubic')
    return f(temp)

class clock_fix_object():
    '''Python class object for the storage of clock fix data.'''
    def __init__(self,curr_receiver,receiver_list,projectDB,scratchWS,figureWS, multipath_filter = False):
        # optional argurments
        self.multipath_filter = multipath_filter
        
        # connect to project database
        conn = sqlite3.connect(projectDB, timeout = 30.0)
        c = conn.cursor()
        
        # identify tags
        self.current_receiver = curr_receiver
        self.current_tag_id = pd.read_sql('SELECT Tag_ID FROM tblReceiver WHERE Rec_ID = "%s"'%(self.current_receiver),con = conn).Tag_ID.values[0]        
        self.receiver_list = receiver_list
        self.projectDB = projectDB
        
        # get study paramters
        self.ref_elev = pd.read_sql("SELECT Ref_Elev FROM tblReceiver WHERE Rec_ID = '%s'"%(self.current_receiver),con = conn).Ref_Elev.values[0]
        
        # get the pulse rate of the current tag
        self.current_pulse_rate = pd.read_sql('SELECT pulseRate FROM tblTag WHERE Tag_ID = "%s"'%(self.current_tag_id), con = conn).pulseRate.values[0]
        
        # get receiver data and transform X and Y 
        self.recDist = pd.DataFrame(columns = ['rec_i','rec_j','dist'])
        recSQL = 'SELECT * FROM tblReceiver WHERE Rec_ID = "%s"'%(receiver_list[0])
        for i in receiver_list[1:]:
            recSQL = recSQL + ' OR Rec_ID = "%s"'%(i)
        self.receivers = pd.read_sql(recSQL, con = conn)
        self.receivers.set_index('Rec_ID',drop = False,inplace = True)
        
        self.master_clock_rec_ID = pd.read_sql('SELECT masterReceiver FROM tblStudyParameters', con = conn).masterReceiver.values[0]
        self.master_clock_tag_ID = self.receivers[self.receivers.Rec_ID == self.master_clock_rec_ID].Tag_ID.values[0] # get the tag_ID for the master clock

        self.master_pulse_rate = pd.read_sql('SELECT pulseRate FROM tblTag WHERE Tag_ID = "%s"'%(self.master_clock_tag_ID), con = conn).pulseRate.values[0] # get the pulse rate for the master clock      
        self.ref_elev = self.receivers[self.receivers.Rec_ID == self.master_clock_rec_ID].Ref_Elev.values[0] # what is the Z elevation in the receiver table referencing?
        
        # get time of transmission
        sql = "SELECT transNo, seconds FROM tblMetronomeFiltered WHERE Rec_ID = '%s' AND Tag_ID = '%s' AND multipath = 0"%(self.master_clock_rec_ID,self.master_clock_tag_ID)
        self.ToT = pd.read_sql_query(sql,con = conn) 
        print ("Returned %s number of rows"%(len(self.ToT)))
        self.ToT.rename(columns = {'seconds':'ToT'},inplace = True)
        self.ToT.set_index('transNo',inplace = True)
        self.ToT.drop_duplicates(inplace = True)
              
        # get timestamped WSEL data and create an interpolator so we can get WSEL at t
        WSELdf = pd.read_sql('SELECT * FROM tblWSEL',con = conn)
        WSELdf['timeStamp'] = pd.to_datetime(WSELdf.timeStamp)
        WSELdf['seconds'] = pd.DatetimeIndex(WSELdf.timeStamp).astype(np.int64)/1.0e9
        self.benchmark_elev = pd.read_sql("SELECT BM_Elev from tblStudyParameters", con = conn).values
        self.elev_units = pd.read_sql('SELECT BM_Elev_Units FROM tblStudyParameters', con = conn).BM_Elev_Units.values[0]
        self.output_units = pd.read_sql('SELECT Output_Units FROM tblStudyParameters', con = conn).Output_Units.values[0]
        if self.elev_units == 'feet' and self.output_units == 'meters':
            WSELdf['WSEL'] = WSELdf.WSEL/3.28084
            self.benchmark_elev = self.benchmark_elev/3.28084
        self.WSELfun = interp1d(WSELdf.seconds,WSELdf.WSEL,kind = 'linear')
        
        # get multipath filtered recapture data from the master clock
        dataSQL = 'SELECT * FROM tblMetronomeFiltered WHERE Rec_ID = "%s" AND Tag_ID = "%s" AND multipath = 0'%(self.current_receiver,self.master_clock_tag_ID)
        self.clock_data = pd.read_sql(dataSQL,con = conn)
        self.clock_data['transNo'] = self.clock_data.transNo

        self.scratchWS = scratchWS
        self.figureWS = figureWS  
        c.close()
        
        # if we are multipath filtering, get training data and train a Gaussian Naive Bayes
        if self.multipath_filter == True:
            self.train_dat = pd.read_sql('SELECT * FROM tblMetronomeFiltered WHERE Rec_ID = "%s"'%(self.current_receiver), con = conn)
            # set indices
            self.train_dat.set_index('transNo',inplace = True,drop = False)
            self.train_dat.sort_index(inplace = True)
            self.train_dat = self.train_dat[self.train_dat.SNR > 0.0]
            # make sure epoch is an integer
            self.train_dat['logAMP'] = np.log10(self.train_dat.Amplitude.values)
            self.train_dat['logNBW'] = np.log10(self.train_dat.NBW.values)
            self.train_dat['logSNR'] = np.log10(self.train_dat.SNR.values)
            self.train_dat.dropna(inplace = True)
            
            primary = self.train_dat[self.train_dat.multipath == 0]
            mutli = self.train_dat[self.train_dat.multipath == 1]
            # 3d plot
            fig = plt.figure(figsize = (6,6))
            ax = fig.add_subplot(111,projection = '3d')
            ax.plot(primary.Amplitude.values, primary.SNR.values, primary.NBW.values,'ko', label = 'primary')
            ax.plot(mutli.Amplitude.values, mutli.SNR.values, mutli.NBW.values,'ro', alpha = 0.2,label = 'multipath')
            ax.legend()
            ax.set_xlabel('Amplitude')
            ax.set_ylabel('Signal to Noise Ratio')
            ax.set_zlabel('Noise in Bandwidth')
            plt.savefig(os.path.join(figureWS,"filter_variables.png"),bbox_inches = 'tight')
            plt.show()
#
#            fuck
            
            # let's make a classifier, if we have training data make a supervised, if not, make an unsupervised 
            # to make a supervised classifier, we need training data       
            train_idx = self.train_dat[self.train_dat.det_rank > 1.0].index.tolist() # training data occurs when we have a primary detection and known multipath - aka, any detection with a detection rank greater than 1
            self.train_dat =  self.train_dat.loc[train_idx]                    # get the indices for these rows 
            #train_dat = []
            if len(self.train_dat) > 0:                 
                X =  self.train_dat[['logSNR','logAMP','logNBW']]              # get predictors
                print (self.train_dat.groupby(['multipath'])[['logSNR','logAMP','logNBW']].describe())
                y =  self.train_dat[['multipath']]                             # get response variable
                self.classifier = nb.GaussianNB()                                # create classifier object
                self.classifier.fit(X,y)                                       # train a classifier        
                # make a prediction on the multipath status of the detection 


def clock_fix(clock_fix_object):
    '''function to fix clocks on a receiver by receiver basis - we will use mutiprocessing
    to speed things up'''
    master_receiver = clock_fix_object.master_clock_rec_ID                                         # get the Rec ID of the master receiver
    current_receiver = clock_fix_object.current_receiver                                                 # get the Rec ID of the current receiver
    master_elev_ref = clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == master_receiver].Ref_Elev.values[0]
    # get master clock recapture data for this receiver 
    receiver_dat = clock_fix_object.clock_data[clock_fix_object.clock_data.Rec_ID == current_receiver]    
    print ("Length of receiver data = %s rows"%(len(receiver_dat)))
    if len(receiver_dat) > 0:
        # calculate the distance between this receiver and the master receiver at time t
        x1 = clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == master_receiver].X_t.values[0]
        y1 = clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == master_receiver].Y_t.values[0]
        z1 = clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == master_receiver].Z_t.values[0]             
        x2 = clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == current_receiver].X_t.values[0]
        y2 = clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == current_receiver].Y_t.values[0]
        z2 = clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == current_receiver].Z_t.values[0]
        # add columns for known receiver locations
        receiver_dat['x1'] = np.repeat(x1,len(receiver_dat))
        receiver_dat['y1'] = np.repeat(y1,len(receiver_dat))
        if master_elev_ref == 'BM':
            receiver_dat['z1'] = np.repeat(z1,len(receiver_dat))
        else:
            t = receiver_dat.seconds.values
            Zt = clock_fix_object.WSELfun(t) - z1
            z1_fix = clock_fix_object.benchmark_elev - Zt
            receiver_dat['z1'] = z1_fix      
        receiver_dat['x2'] = np.repeat(x2,len(receiver_dat))
        receiver_dat['y2'] = np.repeat(y2,len(receiver_dat))
        '''get the Z coordinate of the clock we are fixing.  but be careful, some 
        receivers track with WSEL - need to be careful and calculate a Z relative to 
        the projects benchmark elevation at t if required'''
        if clock_fix_object.ref_elev == 'BM':
            receiver_dat['z2'] = np.repeat(z2,len(receiver_dat))
        else:
            t = receiver_dat.seconds.values
            Zt = clock_fix_object.WSELfun(t) - z2
            z2_fix = clock_fix_object.benchmark_elev - Zt
            receiver_dat['z2'] = z2_fix    
        # calculate distance between receivers at time
        def dist_fun(row):
            from_pos = np.array([row['x1'],row['y1'],row['z1']])
            to_pos = np.array([row['x2'],row['y2'],row['z2']])
            return np.linalg.norm(to_pos - from_pos) 
        receiver_dat['dist'] = receiver_dat.apply(dist_fun,axis = 1)
        receiver_dat = receiver_dat[receiver_dat.SNR > 0.0] 
        receiver_dat.sort_values(by = 'seconds', ascending = True, inplace = True) # sort values by microsecond since epoch
        receiver_dat['lag'] = receiver_dat.seconds.diff()                   # calculate the lag in seconds between the previous detection and the current one
        receiver_dat['leap'] = np.abs(receiver_dat.seconds.diff(-1))                 # calculate the lag in seconds between the next detection and the current one               

        # experience has shown that there are still errors, plot Amplitude vs Noise In Bandwidth 
        if clock_fix_object.multipath_filter == True:
            if len(clock_fix_object.train_dat) > 0:
                # first let's get training data
                receiver_dat['logSNR'] = np.log10(receiver_dat.SNR.values)
                receiver_dat['logAMP'] = np.log10(receiver_dat.Amplitude.values)
                receiver_dat['logNBW'] = np.log10(receiver_dat.NBW.values)
    
                
                receiver_dat['multipath_prediction'] = clock_fix_object.classifier.predict(receiver_dat[['logSNR','logAMP','logNBW']]) # predict classes of unknown detections
                primary = receiver_dat[receiver_dat.multipath_prediction == 0]
                multi = receiver_dat[receiver_dat.multipath_prediction == 1]            
                
                # plot our two groups of detections with results of t-test
                fig = plt.figure(figsize = (6,6))
                ax = fig.add_subplot(111)
    
                # 3d plot
                fig = plt.figure(figsize = (6,6))
                ax = fig.add_subplot(111,projection = '3d')
                ax.plot(primary.logAMP.values, primary.logSNR.values, primary.logNBW.values,'ko', label = 'primary')
                ax.plot(multi.logAMP.values, multi.logSNR.values, multi.logNBW.values,'ro', alpha = 0.2,label = 'multipath')
                ax.legend()
                ax.set_xlabel('Amplitude')
                ax.set_ylabel('Signal to Noise Ratio')
                ax.set_zlabel('Noise in Bandwidth')
                plt.savefig(os.path.join(clock_fix_object.figureWS,"filter_variables_in3space_f%s_t%s.png"%(clock_fix_object.master_clock_rec_ID,clock_fix_object.current_receiver)),bbox_inches = 'tight')
                plt.show()
    
                
    #            # supplot 1, show time dilation 
    #            ax.plot(primary.Amplitude,primary.SNR,'bo',label = 'Primary')
    #            ax.plot(multi.Amplitude,multi.SNR,'ro',label = 'Multipath')
    #            plt.legend()
    #            ax.set_xlabel('Amplitude')
    #            ax.set_ylabel('SNR')
    #            plt.savefig(os.path.join(clock_fix_object.figureWS,'DetectionClasses_f%s_t%s.png'%(clock_fix_object.master_clock_rec_ID,clock_fix_object.current_receiver)),bbox_inches = 'tight')
    #            plt.show()
                
                receiver_dat  = receiver_dat[receiver_dat.multipath_prediction == 0] 
                del primary, multi
            
            
            
        print ("After multipath removal, receiver dat is %s recornds long"%(len(receiver_dat)))
        print (receiver_dat[['transNo','Rec_ID','Tag_ID','seconds','lag','leap']].head(20))
        receiver_dat.dropna(axis = 0, how = 'any', subset = ['transNo'], inplace = True)
        receiver_dat.set_index('transNo',inplace = True)
        receiver_dat = receiver_dat.join(clock_fix_object.ToT,how = 'left') # join dataframe to time of transmission on transmission number
        #print (receiver_dat.head(20))
        receiver_dat.dropna(axis = 0, how = 'any', subset = ['ToT', 'seconds'], inplace = True)
        receiver_dat.reset_index(inplace = True)
        print ("After join, receiver dat %s rows long"%(len(receiver_dat)))

        interpolator = temp_interpolator(clock_fix_object.projectDB,'linear')              # create a temperature interpolator
        
        # calculate time difference of arrival (TDoA) and then change in TDoA
        receiver_dat['TDoA'] = (receiver_dat.seconds - receiver_dat.ToT)       # calculate time difference in arrival (s) between the time of transmission at beacon tag i and time of arrival at receiver j    
        receiver_dat['TDoAlag'] = receiver_dat.TDoA.diff()                     # calculate the lag in seconds between the previous detection and the current one
        
        clock_fix_object.receiver_dat = receiver_dat

        if clock_fix_object.current_receiver != clock_fix_object.master_clock_rec_ID:                                                        # if receiver j isn't attached to beacon i, make a plot and save to csv           
            '''check initial TDoA and quantify translation error.  If the receiving clock
            was ahead of the master clock at the onset of the study, the master receiver will
            appear closer than the current receiver and our distances will be fucky.  
                
            We need to be calculate the expected TDoA given the receiver's position and 
            the current water temperature.  
            '''
            # calculate the temperature
            receiver_dat = clock_fix_object.receiver_dat
            times = receiver_dat.seconds.values
            for i in interpolator:
                avg_C = interpolator[i](times)
            SoS = sos(avg_C) 
            receiver_dat['avg_C'] = avg_C                                      # calculate temperature
            receiver_dat['SoS'] = SoS                                          # write to dataframe

            # expected time of arrival in microseconds, assume we know ToT, dist and SoS exactly 
            receiver_dat['ToA_expected'] = receiver_dat.ToT + receiver_dat.dist/receiver_dat.SoS 
            receiver_dat['TDoA_expected'] = receiver_dat.dist/receiver_dat.SoS
                        
            # calculate the error or residual in time of arrival (measured - expected) 
            receiver_dat['ToA_error'] = receiver_dat.seconds - receiver_dat.ToA_expected # ToA error in seconds
            receiver_dat['TDoA_error'] = receiver_dat.TDoA - receiver_dat.TDoA_expected # TDoA error in seconds

            # fix the time of arrival (measured - error)
            receiver_dat['seconds_fix'] = receiver_dat.seconds - receiver_dat.ToA_error 

            # fix the time difference of arrival (measured + error)
            receiver_dat['TDoA_t'] = receiver_dat.TDoA - receiver_dat.TDoA_error # translated time difference of arrival 
                                                          
            # calculate the mreasued distance difference of arrival 
            receiver_dat['DDoA'] = receiver_dat.SoS * receiver_dat.TDoA
            receiver_dat['deltaDDoA'] = receiver_dat.DDoA.diff() 
            receiver_dat['DDoA_Diff2'] = receiver_dat.deltaDDoA.diff()
            #receiver_dat = receiver_dat[(receiver_dat.deltaDDoA > receiver_dat.deltaDDoA.quantile(0.20)) & (receiver_dat.deltaDDoA < receiver_dat.deltaDDoA.quantile(0.80))]
            #fuck
            #receiver_dat = receiver_dat[(receiver_dat.deltaDDoA == 0)]
            print (receiver_dat[['transNo','seconds','dist','SoS','ToA_error']].head())
            clock_fix_object.receiver_dat = receiver_dat                       # write clock fix to object

            fig = plt.figure(figsize = (6,4))
            ax = fig.add_subplot(111)
            
            # make a plot showing how bad of a problem it is
            gs = gridspec.GridSpec(4,4)
            gs.update(hspace = 0.5)
            fig = plt.figure(figsize = (6,7))
            # supplot 1, show time dilation 
            ax1 = plt.subplot(gs[0:2,:])
            ax1.plot(receiver_dat.transNo.values,receiver_dat.DDoA.values,'ro',label = 'recorded')            
            ax1.set_xlabel('Transmission Number')
            ax1.set_ylabel('Distance Difference of Arrival (m)')
            ax1.set_title('Effects of Clock Drift: \n Difference between ToA at Receiver %s and ToT from Receiver %s'%(clock_fix_object.current_receiver,clock_fix_object.master_clock_rec_ID))
            # subplot 2, show position of receivers
            ax2 = plt.subplot(gs[2:4,:])
            ax2.plot(clock_fix_object.receivers.X_t.values,clock_fix_object.receivers.Y_t.values,'bo',label = 'Receiver')
            ax2.plot(clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == clock_fix_object.master_clock_rec_ID].X_t.values[0],clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == clock_fix_object.master_clock_rec_ID].Y_t.values[0],'ro',label = 'Origin Receiver')
            ax2.plot(clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == clock_fix_object.current_receiver].X_t.values[0],clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == clock_fix_object.current_receiver].Y_t.values[0],'go',label = 'Test Receiver')
            ax2.set_xlabel('Easting (m) \n Translation to UTM Zone 10 North NAD1983: \n add %s m E and %s m N'%(clock_fix_object.receivers.X.mean(),clock_fix_object.receivers.Y.mean()))
            ax2.set_ylabel('Northing (m)')
            ax2.text(clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == clock_fix_object.master_clock_rec_ID].X_t.values[0]+0.25,clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == clock_fix_object.master_clock_rec_ID].Y_t.values[0]+0.25,'%s'%(clock_fix_object.master_clock_rec_ID))
            ax2.text(clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == clock_fix_object.current_receiver].X_t.values[0]+0.25,clock_fix_object.receivers[clock_fix_object.receivers.Rec_ID == clock_fix_object.current_receiver].Y_t.values[0]+0.25,'%s'%(clock_fix_object.current_receiver))
            ax2.legend(title = 'Stationary Receivers')
            #plt.show()
            plt.savefig(os.path.join(clock_fix_object.figureWS,'ClockDrift_f%s_t%s.png'%(clock_fix_object.master_clock_rec_ID,clock_fix_object.current_receiver)),bbox_inches = 'tight')        
            
            # let's see if we can fix that error 
            clock_fix_object.receiver_dat = receiver_dat 
            receiver_dat.sort_values(by = 'seconds',inplace = True)
            # Fit picewise linear to error line:
            residual = interp1d(receiver_dat.seconds.values,receiver_dat.ToA_error.values,kind = 'linear',bounds_error = False, fill_value = 9999)  # create a function for the residual at time of arrival  
            receiver_dat['seconds_residual_predicted'] = residual(receiver_dat.seconds)   # predict the residual and visually confirm with plot            
            receiver_dat['prev_seconds'] = receiver_dat.seconds.shift()
            receiver_dat['prev_error'] = receiver_dat.ToA_error.shift()
            receiver_dat['beta1'] = receiver_dat.prev_error
            receiver_dat['beta2'] = (receiver_dat.ToA_error - receiver_dat.prev_error)/(receiver_dat.seconds - receiver_dat.prev_seconds)                      
                        
            conn = sqlite3.connect(clock_fix_object.projectDB, timeout = 30.0)
            c = conn.cursor()
            
            # fix the time of arrival at receiver j
            sql = "SELECT * FROM tblDetectionRaw WHERE Rec_ID = '%s'"%(clock_fix_object.current_receiver)
            curr_rec_dat = pd.read_sql_query(sql,conn)
            curr_rec_dat['timeDiff'] = np.zeros(len(curr_rec_dat))
            curr_rec_dat.sort_values(by = 'seconds',inplace = True)
            curr_rec_dat['seconds_residual'] = residual(curr_rec_dat.seconds)
            curr_rec_dat = curr_rec_dat[curr_rec_dat.seconds_residual != 9999]
            curr_rec_dat['seconds_fix'] = curr_rec_dat.seconds - curr_rec_dat.seconds_residual
            firstDet = curr_rec_dat.seconds_fix.min()                        # find the first ever detection of this tag
            curr_rec_dat['timeDiff'] = curr_rec_dat.seconds_fix.values - firstDet # calculate the difference between the time and the first detection

            times = curr_rec_dat.seconds_fix.values

            avg_C = []
            temps = []
            for j in interpolator:
                avg_C = interpolator[j](times)
            SoS = sos(avg_C)
            curr_rec_dat['avg_C'] = avg_C # calculate temperature
            curr_rec_dat['SoS'] = SoS                                          # write to dataframe            
            # export data to csv
            curr_rec_dat.to_csv(os.path.join(clock_fix_object.scratchWS,'receiver_%s_epoch_fix.csv'%(clock_fix_object.current_receiver)), index = False, float_format='%.6f')
            receiver_dat.to_csv(os.path.join(clock_fix_object.figureWS,'receiver_%s_clock_fix.csv'%(clock_fix_object.current_receiver)), float_format='%.6f')  
            
            # Make a plot, show measured error vs interpolated error 
            fig = plt.figure(figsize = (6,4))
            ax = fig.add_subplot(111)
            ax.plot(receiver_dat.seconds_fix,receiver_dat.ToA_error,'b-', label = 'measured error')
            ax.plot(curr_rec_dat.seconds_fix,curr_rec_dat.seconds_residual,'r--',label = 'interpolated error')
            plt.legend()
            ax.set_ylabel('Residual')
            ax.set_xlabel('transmission')
            #plt.show() 
            plt.savefig(os.path.join(clock_fix_object.figureWS,'measured_vs_interpolated_error_check_%s_t%s.png'%(clock_fix_object.master_clock_rec_ID,clock_fix_object.current_receiver)),bbox_inches = 'tight')

            # Make a plot, show measured v expected TDoA
            fig = plt.figure(figsize = (6,4))
            ax = fig.add_subplot(111)
            ax.plot(receiver_dat.transNo,receiver_dat.TDoA,'b-', label = 'measured TDoA')
            ax.plot(receiver_dat.transNo,receiver_dat.TDoA_expected,'r-',label = 'expected TDoA')
            plt.legend()
            ax.set_ylabel('Residual')
            ax.set_xlabel('transmission')
            #plt.show() 
            plt.savefig(os.path.join(clock_fix_object.figureWS,'measured_vs_expected_TDoA_%s_t%s.png'%(clock_fix_object.master_clock_rec_ID,clock_fix_object.current_receiver)),bbox_inches = 'tight')

            
        else:
            conn = sqlite3.connect(clock_fix_object.projectDB, timeout = 30.0)
            c = conn.cursor()
            # fix the time of arrival at receiver j
            sql = "SELECT * FROM tblDetectionRaw WHERE Rec_ID = '%s'"%(clock_fix_object.current_receiver)
            curr_rec_dat = pd.read_sql_query(sql,conn) 
            curr_rec_dat['timeDiff'] = np.zeros(len(curr_rec_dat))
            c.close()
            curr_rec_dat.sort_values(by = 'seconds',inplace = True)
            curr_rec_dat['seconds_residual'] = np.repeat(0,len(curr_rec_dat))
            curr_rec_dat['seconds_fix'] = curr_rec_dat.seconds - curr_rec_dat.seconds_residual
            
            times = curr_rec_dat.seconds_fix.values

            avg_C = []
            temps = []
            for j in interpolator:
                avg_C = interpolator[j](times)
            SoS = sos(avg_C)        
            curr_rec_dat['avg_C'] = avg_C                                      # calculate temperature
            curr_rec_dat['SoS'] = SoS                                          # write to dataframe   
            # export data to csv
            curr_rec_dat.to_csv(os.path.join(clock_fix_object.scratchWS,'receiver_%s_epoch_fix.csv'%(clock_fix_object.current_receiver)), index = False, float_format='%.6f')
            receiver_dat.to_csv(os.path.join(clock_fix_object.figureWS,'receiver_%s_clock_fix.csv'%(clock_fix_object.current_receiver)), float_format='%.6f')  

    else:
        print ("No data for receiver %s check inputs"%(clock_fix_object.current_receiver))

def epoch_fix_data_management(inputWS,projectDB):
    # As soon as I figure out how to do this function is moot.
    files = os.listdir(inputWS)
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    for f in files:
        dat = pd.read_csv(os.path.join(inputWS,f))#,dtype = {"detHist":str})
        dat.to_sql('tblDetectionClockFixed',con = conn,index = False, if_exists = 'append', chunksize = 1000)
        os.remove(os.path.join(inputWS,f))
        del dat
    # create an index on tblMetronomeUnfiltered 
    c.execute('''CREATE INDEX idx_combined_clock_fix ON tblDetectionClockFixed (Rec_ID, Tag_ID, seconds_fix)''')
    conn.commit()
    c.close()                
    c.close()      


class position():
    '''a python class object to hold acoustic data for a single tag to be processed 
    for positioning and a series of methods designed to coordinate fish using 
    time difference of arrival methods.
    
    A method to implement Daniel Deng's exact solution and an ordinary least squares 
    statistical method.'''
    def __init__(self,tag,resolved_clocks,projectDB,outputWS,figureWS):
        # set important variables and identifiers
        self.tag = tag                                                         # current tag
        self.resolved_clocks = resolved_clocks                                 # not all clocks can be fixed, this should only be those appropriate for modeling 
        self.projectDB = projectDB                                             # the project database, aka dbDir
        self.outputWS = outputWS                                               # where are we going to put the t,X,Y,Z table?
        self.figureWS = figureWS                                               # everyone loves a pretty picture, let's save it for posterity

        # connect to project database and get this tag's data at each resolved clock        
        conn = sqlite3.connect(projectDB, timeout = 30.0)
        c = conn.cursor()
        self.tag_data = pd.DataFrame()
        for i in self.resolved_clocks:
            dat = pd.read_sql('SELECT * FROM tblDetectionFilterSecondary WHERE Tag_ID = "%s" AND Rec_ID = "%s" AND multipath = 0 AND multipath_prediction = 0'%(self.tag,i), con = conn)
            self.tag_data = self.tag_data.append(dat)
            
        # build an ephemeris
        self.recDist = pd.DataFrame(columns = ['rec_i','rec_j','dist'])
        recSQL = 'SELECT * FROM tblReceiver WHERE Rec_ID = "%s"'%(self.resolved_clocks[0])
        for i in resolved_clocks[1:]:
            recSQL = recSQL + ' OR Rec_ID = "%s"'%(i)
        self.ephemeris = pd.read_sql(recSQL, con = conn)
        self.ephemeris.set_index('Rec_ID',drop = False,inplace = True)
        self.convex_hull = ConvexHull(np.array(self.ephemeris[['X_t','Y_t','Z_t']]))
                                                                
        # get timestamped WSEL data and create an interpolator so we can get WSEL at t
        WSELdf = pd.read_sql('SELECT * FROM tblWSEL',con = conn)
        WSELdf.dropna(inplace = True)
        WSELdf['timeStamp'] = pd.to_datetime(WSELdf.timeStamp)
        WSELdf['seconds'] = WSELdf.timeStamp.astype(np.int64)/1.0e9
        print ("max timestamp is %s"%(WSELdf.seconds.max()))
        self.benchmark_elev = pd.read_sql("SELECT BM_Elev from tblStudyParameters", con = conn).values[0]
        self.elev_units = pd.read_sql('SELECT BM_Elev_Units FROM tblStudyParameters', con = conn).BM_Elev_Units.values[0]
        self.output_units = pd.read_sql('SELECT Output_Units FROM tblStudyParameters', con = conn).Output_Units.values[0]
        #if self.elev_units == 'feet' and self.output_units == 'meters':
            
        WSELdf['WSEL'] = WSELdf.WSEL/3.28084
            #self.benchmark_elev = self.benchmark_elev/3.28084
        self.WSELfun = interp1d(WSELdf.seconds,WSELdf.WSEL,kind = 'linear')


        self.tagType = pd.read_sql('SELECT TagType FROM tblTag WHERE Tag_ID = "%s"'%(tag),con = conn).TagType.values[0]
        self.pulseRate = pd.read_sql('SELECT pulseRate FROM tblTag WHERE Tag_ID = "%s"'%(tag),con = conn).pulseRate.values[0]
        
        # get temperature at t using our temperature interpolator
        self.interpolator = temp_interpolator(self.projectDB,'linear')      # create a temperature interpolator

        c.close()
    
    def Deng(self):
        def point_in_hull(point,hull):
            tolerance = 1e-12
            return all(
                (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
                for eq in hull.equations)



        Solution_Cols = ['transNo','r0','r1','r2','r3','X','Y','Z','T01','ToA','comment','in_hull']
        
        SolutionA = pd.DataFrame(columns = Solution_Cols)
        SolutionB = pd.DataFrame(columns = Solution_Cols)
        self.tag_data.sort_values(by = 'seconds_fix', axis = 0, ascending = True, inplace = True)
        tSteps = self.tag_data.transNo.unique()                                       # identify the unique transmissions
        for j in sorted(tSteps):
            tDat = self.tag_data[self.tag_data.transNo == j]                   # get data associated with this transmission
            tDat['timeStampFix'] = pd.to_datetime(tDat.seconds_fix, unit = 's')
            tDat['timeStampOriginal'] = pd.to_datetime(tDat.seconds, unit = 's')
            if len(tDat) >= 4: # in a perfect world, we have enogh receivers with enough observations to calculate a solution, we need at least 4
                #print tDat[['transNoFix','Rec_ID','timeStampOriginal','timeStampFix']]
                #fuck
                # find the reference receiver - aka one with first recapture - time difference of arrival and all 
                tDat.sort_values(by = 'seconds_fix', axis = 0, ascending = True, inplace = True)
                ref = tDat.iloc[0].Rec_ID                                      # our reference time is the first time of arrival
                r1 = tDat.iloc[1].Rec_ID
                r2 = tDat.iloc[2].Rec_ID
                r3 = tDat.iloc[3].Rec_ID                                       # we only need 4 for Deng's exact solution, this will do
                t_ref = tDat.seconds_fix.iloc[0]                                 # time of arrival 1
                t1 = tDat.seconds_fix.iloc[1]
                t2 = tDat.seconds_fix.iloc[2]
                t3 = tDat.seconds_fix.iloc[3]                               # time of arrival 4
                # get positional data of receivers
                def z_at_t(t,Rec_ID):
                    elev_ref = self.ephemeris[self.ephemeris.Rec_ID ==Rec_ID].Ref_Elev.values[0]
                    if elev_ref == 'BM':
                        return self.ephemeris[self.ephemeris.Rec_ID ==Rec_ID].Z_t.values[0]
                    else:
                        Zt = self.WSELfun(t) - self.ephemeris[self.ephemeris.Rec_ID ==Rec_ID].Z_t.values[0]
                        t_Z = self.benchmark_elev - Zt
                        return t_Z.tolist()[0]
                        
                r0Pos = np.array([self.ephemeris[self.ephemeris.Rec_ID == ref].X_t.values[0],
                            self.ephemeris[self.ephemeris.Rec_ID == ref].Y_t.values[0],
                            z_at_t(t_ref,ref)]) #(X,Y,Z of reciever 0)
                print ("The position of receiver 0 is %s"%(r0Pos))
                self.z_translation = r0Pos[2]
                r1Pos = np.array([self.ephemeris[self.ephemeris.Rec_ID == r1].X_t.values[0],
                            self.ephemeris[self.ephemeris.Rec_ID == r1].Y_t.values[0],
                            z_at_t(t1,r1)]) #(X,Y,Z of reciever 1)
                print ("The position of receiver 1 is %s"%(r1Pos))      
                r2Pos = np.array([self.ephemeris[self.ephemeris.Rec_ID == r2].X_t.values[0],
                            self.ephemeris[self.ephemeris.Rec_ID == r2].Y_t.values[0],
                            z_at_t(t2,r2)]) #(X,Y,Z of reciever 2)
                print ("The position of receiver 2 is %s"%(r2Pos))                            
                r3Pos = np.array([self.ephemeris[self.ephemeris.Rec_ID == r3].X_t.values[0],
                            self.ephemeris[self.ephemeris.Rec_ID == r3].Y_t.values[0],
                            z_at_t(t3,r3)]) #(X,Y,Z of reciever 3)         
                print ("The position of receiver 3 is %s"%(r3Pos))
                                          
                # create matrices, equation 3a Deng 2011
                #R = np.matrix(np.vstack(((r1Pos - r0Pos),(r2Pos - r0Pos),(r3Pos - r0Pos))))
                R = np.matrix(np.array([[r1Pos[0]-r0Pos[0],r2Pos[0]-r0Pos[0],r3Pos[0]-r0Pos[0]],[r1Pos[1]-r0Pos[1],r2Pos[1]-r0Pos[1],r3Pos[1]-r0Pos[1]],[r1Pos[2]-r0Pos[2],r2Pos[2]-r0Pos[2],r3Pos[2]-r0Pos[2]]]))
                print ("Posisition Matrix R \n %s"%(R))
                # vertical matrix of time of arrival differences (TOADs) between receiver i and reference receiver #1
                tdoa_1 = np.round(t1 - t_ref,6)                                    # difference in time of arrival between current receiver and reference receiver
                tdoa_2 = np.round(t2 - t_ref,6)
                tdoa_3 = np.round(t3 - t_ref,6)              
                t = np.matrix(np.vstack((tdoa_1,tdoa_2,tdoa_3)))               # column matrix of tdoa's                                                     
                print ("Print T matrix \n %s"%(t))           
                # calculate speed of sound 
                temps = []
                for i in self.interpolator:                                    # for each temperature interpolator....
                    temps.append(self.interpolator[i](t_ref))                  # get the temperature at time and append to temps list
                avg_C = np.nanmean(temps)                                      # calculate the mean and ignore nan
                print ("Current temperature = %s" %(avg_C))
                SoS = sos(avg_C)                                               # calculate the speed of sound at the current temperature
                print  ("Current Speed of Sound = %s"%(SoS))
                
                # vertical matrix of b, equation 3a Deng 2011
                b1 = np.linalg.norm(r1Pos-r0Pos)**2 - (SoS**2 * tdoa_1**2)
                b2 = np.linalg.norm(r2Pos-r0Pos)**2 - (SoS**2 * tdoa_2**2) 
                b3 = np.linalg.norm(r3Pos-r0Pos)**2 - (SoS**2 * tdoa_3**2)  
                b = np.matrix(np.vstack((b1,b2,b3)))
                print ("B matrix \n %s"%(b))
                
                # solve for T_0 equations 5 and 5a Deng 2011
                try:
                    a = SoS**4 * t.T * R.I * R.T.I * t - SoS**2    
                    p = -0.5 * SoS**2 * t.T * R.I * R.T.I * b 
                    q = 0.25 * b.T * R.I * R.T.I * b
                    
                    # Solve for ToA
                    T_0a = (-p + np.sqrt(p**2 - a*q))/a
                    T_0b = (-p - np.sqrt(p**2 - a*q))/a
                    
                    # solve for S = positon of source with equation 4 from Deng 2011
                    if np.sign(T_0a) > 0:
                        S1a = R.I.T * (0.5 * b - SoS**2 * t * T_0a)
                        point = np.array([r0Pos[0] + S1a.item(0),r0Pos[1] + S1a.item(1),r0Pos[2] + S1a.item(2)])
                        in_hull = point_in_hull(point,self.convex_hull)
                        row = pd.DataFrame(np.array([[j,ref,r1,r2,r3,
                                                      r0Pos[0] + S1a.item(0),
                                                    r0Pos[1] + S1a.item(1),
                                                    r0Pos[2] + S1a.item(2),
                                                    T_0a.item(0),
                                                    tDat.seconds_fix.values[0],'solution found',in_hull]]),columns = Solution_Cols)   
                        SolutionA = SolutionA.append(row)
                        del row            
                        print ("Solution Found for fish %s at transmission %s"%(i,j))
                        print ("Fish at %s,%s,%s"%(S1a[0]+r0Pos[0],S1a[1]+r0Pos[1],S1a[2]+r0Pos[2]))  
                        
                    else:
                        print ("No solution A found time step %s"%(j))
                        row = pd.DataFrame(np.array([[j,ref,r1,r2,r3,'','','','','','negative time of arrival - no soluiton','']]), columns = Solution_Cols)
                        SolutionA = SolutionA.append(row)
                        
                    if np.sign(T_0b) > 0:
                        S1b = R.I.T * (0.5 * b - SoS**2 * t * T_0b)
                        point = np.array([r0Pos[0] + S1b.item(0),r0Pos[1] + S1a.item(1),r0Pos[2] + S1a.item(2)])
                        in_hull = point_in_hull(point,self.convex_hull)
                        row = pd.DataFrame(np.array([[j,ref,r1,r2,r3,
                                                      r0Pos[0] + S1b.item(0),
                                                    r0Pos[1] + S1b.item(1),
                                                    r0Pos[2] + S1b.item(2),
                                                    T_0b.item(0),
                                                    tDat.seconds_fix.values[0],'solution found',in_hull]]),columns = Solution_Cols)
                        SolutionB = SolutionB.append(row)
                        del row
                        print ("Solution Found for fish %s at transmission %s"%(i,j))
                        print ("Fish at %s,%s,%s"%(S1b[0]+r0Pos[0],S1b[1]+r0Pos[1],S1b[2]+r0Pos[2]))
                    else:
                        print ("No solution B found time step %s"%(j))
                        row = pd.DataFrame(np.array([[j,ref,r1,r2,r3,'','','','','','negative time of arrival - no soluiton','']]), columns = Solution_Cols)
                        SolutionA = SolutionA.append(row)
                except:
                    print ("Singular matrix encountered, no solution at transmission %s"%(j))
                    row = pd.DataFrame(np.array([[j,ref,r1,r2,r3,'','','','','','singular matrix encountered - no soluiton','']]), columns = Solution_Cols)
                    SolutionA = SolutionA.append(row)

            else:
                print ("Not enough receivers for a solution at time step %s"%(j))
                row = pd.DataFrame(np.array([[j,'','','','','','','','','','not enough receivers for solution','']]), columns = Solution_Cols)
                SolutionA = SolutionA.append(row)
                SolutionB = SolutionB.append(row)                
            
        def distF(row):
            pos1 = np.asarray(row['pos'])
            pos2 = np.asarray(row['nextPos'])
            return np.linalg.norm(pos1-pos2)
        


        self.DengSolutionA_unfiltered = SolutionA
        self.DengSolutionB_unfiltered = SolutionB 
                       
        SolutionA.to_csv(os.path.join(self.outputWS,"%s_solutionA.csv"%(self.tag)))
        SolutionB.to_csv(os.path.join(self.outputWS,"%s_solutionB.csv"%(self.tag)))
#            SolutionA3.to_csv(os.path.join(outputWS,'Production','Files',"%s_solutionA_filtered.csv"%(i)))
#            SolutionB3.to_csv(os.path.join(outputWS,'Production','Files',"%s_solutionB_filtered.csv"%(i)))
    def trajectory_plot_Deng(self, hull_filter = False):

        def distB(row):
            pos1 = np.asarray(row['pos'])
            pos2 = np.asarray(row['prevPos'])
            return np.linalg.norm(pos1-pos2)
        # get solution A
        solA = self.DengSolutionA_unfiltered[self.DengSolutionA_unfiltered.comment == 'solution found']
        solA['X'] = solA.X.astype(np.float32)
        solA['Y'] = solA.Y.astype(np.float32)
        solA['Z'] = solA.Z.astype(np.float32)
        solA['ToA'] = solA.ToA.astype(np.float32)        

        solA = solA[(solA.X > -50) & (solA.X < 50)]
        solA = solA[(solA.Y > -50) & (solA.Y < 50)] 

        
        # get solution B
        solB = self.DengSolutionB_unfiltered[self.DengSolutionB_unfiltered.comment == 'solution found']
        solB['X'] = solB.X.astype(np.float32)
        solB['Y'] = solB.Y.astype(np.float32)
        solB['Z'] = solB.Z.astype(np.float32)
        solB['ToA'] = solB.ToA.astype(np.float32)                   

        solB = solB[(solB.X > -50) & (solB.X < 50)]
        solB = solB[(solB.Y > -50) & (solB.Y < 50)] 

        if hull_filter == True:
            solA = solA[solA.in_hull == True]
            solB = solB[solB.in_hull == True]
                                                                   
        fig = plt.figure()
        ax = fig.add_subplot(111,projection = '3d')
        ax.scatter(self.ephemeris.X_t.values,self.ephemeris.Y_t.values,self.ephemeris.Z_t.values, c = 'blue')
        if len(solB) > 0:
            ax.plot(solB.X.values,solB.Y.values,solB.Z.values, c = 'magenta')    # true data
            ax.scatter(solB.X.values[0],solB.Y.values[0],solB.Z.values[0],c = 'green')#, s = 12)
            ax.scatter(solB.X.values[-1],solB.Y.values[-1],solB.Z.values[-1],c = 'red')# ,s = 12)

        ax.set_xlim(-50,50)
        ax.set_ylim(-50,50)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
                
        plt.show()        
       
def positions_data_management(pos_type,inputWS,projectDB):
    # As soon as I figure out how to do this function is moot.
    files = os.listdir(inputWS)
    conn = sqlite3.connect(projectDB)
    c = conn.cursor()
    if pos_type == 'Deng':
        for f in files:
            solution = f[-5:-4]
            fishy = f[0:4]
            dat = pd.read_csv(os.path.join(inputWS,f))#,dtype = {"detHist":str})
            #dat.drop(labels = ['in_hull','Unnamed: 0'], axis = 1, inplace = True)
            dat['solution'] = np.repeat(solution,len(dat))
            dat['Tag_ID'] = np.repeat(fishy,len(dat))
            dat.to_sql('tblPositions_Deng',con = conn,index = False, if_exists = 'append', chunksize = 1000)
            #os.remove(os.path.join(inputWS,f))
            del dat
        #c.execute('''CREATE INDEX idx_combined_solution ON tblPositions_Deng (Tag_ID, solution, transNo)''')

    # create an index on tblMetronomeUnfiltered 
    conn.commit()
    c.close()                
     
class kernels():
    '''python class object to construct kernel utilization distributions to improve 
    estimates of space use in aquatic animals.  See Simpfendorder et al. 2012'''
    
    def __init__(self,pos_type,projectDB,outputWS,tag_ID = None,):
        '''initialize a kernel use density object.  The only thing we need is the 
        type of position (Deng or Least Squares) and a link to the project database'''
        self.projectDB = projectDB
        self.tag_ID = tag_ID
        self.outputWS = outputWS


        
        if pos_type == 'Deng':
            # get this fish's data
            conn = sqlite3.connect(projectDB)
            c = conn.cursor()
            if tag_ID != None:
                sql = 'SELECT * FROM tblPositions_Deng WHERE Tag_ID == "%s"'%(tag_ID)
                self.dat = pd.read_sql(sql, con = conn)
                self.dat = self.dat[(self.dat.comment == 'solution found') & (self.dat.solution == 'B')] # just get positions
            else:
                sql = 'SELECT * FROM tblPositions_Deng'
                self.dat = pd.read_sql(sql, con = conn)
                self.dat = self.dat[(self.dat.comment == 'solution found') & (self.dat.solution == 'B')] # just get positions
                self.dat = self.dat.iloc[::10, :]
            # create an epehmeris - we don't want densities outside of the convex hull of the receivers
            recSQL = 'SELECT * FROM tblReceiver'
            self.ephemeris = pd.read_sql(recSQL, con = conn)
            self.ephemeris.set_index('Rec_ID',drop = False,inplace = True) 
            
            c.close()
            pos = self.dat[['X','Y','Z']].iloc[::5, :]                         # extract position array  - every 5th
            X = pos.X.values
            Y = pos.Y.values
            Z = pos.Z.values
            pos_arr = np.vstack([X,Y,Z])
            
            # calculate datetime and get grouping variables
            self.dat.datetime = pd.to_datetime(self.dat.ToA)
            self.dat['hour'] = pd.to_datetime(self.dat.ToA).dt.hour
            self.dat['daytime'] = self.dat.hour.apply(lambda x: True if x >= 6 and x < 18 else False)
            
            print ('length of position array %s records'%(len(pos)))

            self.xmin = self.ephemeris.X_t.min()                                        # get the mins and maxs - we don't want to extrapolate 
            self.ymin = self.ephemeris.Y_t.min()
            self.zmin = self.ephemeris.Z_t.min()
            self.xmax = self.ephemeris.X_t.max()
            self.ymax = self.ephemeris.Y_t.max()
            self.zmax = self.ephemeris.Z_t.max()
            
            d = pos_arr.shape[0]
            n = pos_arr.shape[1]
            bw = (n * (d + 2)/ 4.)**(-1. / (d + 4)) # silverman
            #bw = n**(-1./(d + 4))                   # scott
            # create a kernel density object
            
            print ('fitting kernel density estimate')
            
            self.kde = KernelDensity(bandwidth = bw, metric = 'euclidean', kernel = 'gaussian', algorithm = 'ball_tree')
            # fit a kernel density object to the fish's position data
            self.kde.fit(pos_arr.T)
            # create a test grid at the extent of our fish's positions
            xi, yi, zi= np.mgrid[self.xmin:self.xmax:50j, self.ymin:self.ymax:50j,self.zmin:self.zmax:50j]
            self.coords = np.vstack([xi.ravel(), yi.ravel(), zi.ravel()])
            # evaluate the density at each point in our test grid
            self.density = np.reshape(np.exp(self.kde.score_samples(self.coords.T)), xi.shape)
            self.density_norm = self.density / self.density.flatten().max()
            

            
            
        print ("kernel density estimate complete, proceed to plotting")
    def plot(self):
        xi, yi, zi= np.mgrid[self.xmin:self.xmax:50j, self.ymin:self.ymax:50j,self.zmin:self.zmax:50j]
        self.fig = go.Figure(data=go.Volume(
            x=xi.flatten(),
            y=yi.flatten(),
            z=zi.flatten(),
            value=self.density_norm.flatten(),
            #isomin = np.percentile(self.density.flatten(),0.2),
            #isomax = np.percentile(self.density.flatten(),0.8),
            isomin = 0.025,
            opacity=0.1, # needs to be small to see through all surfaces
            surface_count=15, # needs to be a large number for good volume rendering
            ))
        self.fig.write_html(os.path.join(self.outputWS,'%s.html'%(self.tag_ID)),auto_open = True)  
        #self.fig.write_image(os.path.join(self.outputWS,"%s.png"%(self.tag_ID)))
        

        
    def kda(self):
        # use skkda package to perform a kernel discriminant analysis
        print ('performing kernel discriminant analysis')
        self.kda = skkda.base.KernelDiscriminantAnalysis(lmb = 0.001,kernel = 'rbf',degree = 3, gamma = None,coef0 = 1)
        dat2 = self.dat.iloc[::5, :]
        print (dat2.head())
        print (dat2[['daytime']].to_numpy())
        print (dat2[['X','Y','Z']].to_numpy())
        self.kda.fit(X = dat2[['X','Y','Z']].to_numpy(),y = dat2[['daytime']].to_numpy())
        self.predict = self.kda.transform(self.coords.T)
        # plot interpolated scatter plot and color according to value
        xi, yi, zi= np.mgrid[self.xmin:self.xmax:50j, self.ymin:self.ymax:50j,self.zmin:self.zmax:50j]
   
        self.fig = go.Figure(data = go.Scatter3d(
            x = xi.flatten(),
            y = yi.flatten(),
            z = zi.flatten(),
            mode = 'markers',
            marker = dict(
                    size = 5,
                    color = self.predict.flatten(),
                    opacity = 0.5
                    )
                ))
        self.fig.write_html(os.path.join(self.outputWS,'predicted_scatter.html'), auto_open = True)



























                    
