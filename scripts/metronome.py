'''Script Intent: Interface with Kleinschmidt's Proprietary Acoustic Telemetry
Data Management and Positioning Software to perform multipath filtering.

Script utilizes multiprocessing, make sure the number of processes is not greater
than the number of cores you have on your computer.

Script Author: KPN'''

# Import Modules:
import os
import jsats3d
import pandas as pd
import sqlite3
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import time

ts = time.time()
outputWS = r"J:\3870\013\Calcs\Paper\Output"
inputWS = r"J:\3870\013\Calcs\Paper\Data"
dbName = 'cowlitz_2018_paper.db'
dbDir = os.path.join(inputWS,dbName)
#dbDir = r"C:\Users\Kevin Nebiolo\Desktop\cowlitz_tagDrag.db"
print ("Starting Beacon Tag Transmission Enumeration")
metronome = 'FF75'
#get a list of recievers to iterate through
conn = sqlite3.connect(dbDir)
c = conn.cursor() 
receivers = pd.read_sql('SELECT Rec_ID FROM tblReceiver WHERE Tag_ID != "%s"'%(metronome),con = conn).Rec_ID.values

print ("Creating an epoch data object for the study's metronome, tag ID = %s"%(metronome))
metronome_dat = jsats3d.beacon_epoch(metronome,dbDir,outputWS)                    # create an epoch data object for this beacon tag
print ("Enumerating Transmission Numbers at the Host Receiver")
metronome_dat.host_receiver_enumeration()                                          # enumerate the transmission numbers at the host receiver
print ("Enumerating Transmission Numbers at Adjacent Receivers")
metronome_dat.adjacent_receiver_enumeration()                                      # enumerate metronome transmission on the adjacent receivers
print ("Creating multipath data object for metronome data")
metronome_multipath = jsats3d.multipath_data_object(metronome,dbDir,outputWS,metronome = True) 
print ("Multipath filter metronome multipath data object")
jsats3d.multipath_2(metronome_multipath)


print ("All tags processed, proceed to clock fixing")
print ("Metronome filtering took %s seconds to compile"%(round(time.time() - ts,4))) 
    
    