'''Script Intent: Interface with Kleinschmidt's Proprietary Acoustic Telemetry
Data Management and Positioning Software to perform multipath filtering.

Script utilizes multiprocessing, make sure the number of processes is not greater
than the number of cores you have on your computer.

Script Author: KPN'''

# Import Modules:
import os
import jsats3d
import sqlite3
import warnings
warnings.filterwarnings('ignore')

outputWS = r"C:\Users\knebiolo\Desktop\jsats_Notebook_Test\Output"
dbName = 'jsats_notebook_test.db'
dbDir = os.path.join(outputWS,dbName)

# get list of tags
tags = ['FF75']
      
# for i in tags:
#     obj = jsats3d.multipath_data_object(i,dbDir,outputWS)                 # for every tag, make a multipath data object 
#     jsats3d.multipath_2(obj)                                                        # apply primary filter
#     print("Primary filter applied to tag %s"%(i))

# jsats3d.multipath_data_management(outputWS,dbDir,primary = True)          # add data to database

# apply secondary filter
for i in tags:
    print ("Secondary mulitpath filter applied to tag %s"%(i))
    jsats3d.mulitpath_classifier(i,dbDir,outputWS,beacon = True,method = "KNN")          # apply secondary filter

#jsats3d.multipath_data_management(outputWS,dbDir,primary = False)         # add data to database

print ("All tags processed, proceed to positioning")


    
