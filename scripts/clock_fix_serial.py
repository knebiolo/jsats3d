'''Script Intent: Interface with Kleinschmidt's Proprietary Acoustic Telemetry
Data Management and Positioning Software to identify and fix clock drift.

Script utilizes multiprocessing, make sure the number of processes is not greater
than the number of cores you have on your computer.

Script Author: KPN'''

# Import Modules:
import os
import jsats3d
import warnings
warnings.filterwarnings('ignore')
import time

ts = time.time()
outputWS = r"C:\Users\knebiolo\Desktop\jsats with DBSCAN\Output\Scratch"
figureWS = r"C:\Users\knebiolo\Desktop\jsats with DBSCAN\Output\Figures"
inputWS = r"C:\Users\knebiolo\Desktop\jsats with DBSCAN\Output"
dbName = 'cowlitz_test.db'
dbDir = os.path.join(inputWS,dbName)
#dbDir = r"C:\Users\Kevin Nebiolo\Desktop\cowlitz_tagDrag.db"

recList = ['R08']

analysisRecs = ['R04','R05','R06','R07','R08','R09']

# create a clock fix data objects

print ("Start processing receivers")
for i in recList:
    clock_fix_object = jsats3d.clock_fix_object(i,analysisRecs,dbDir,outputWS,figureWS) 
    print ("created preliminary data objects for receiver %s"%(i))
    jsats3d.clock_fix(clock_fix_object)
    print ("Finished Processing Receiver %s"%(i))
    
#jsats3d.epoch_fix_data_management(outputWS,dbDir)

print ("All receivers processed, proceed to positioning")
print ("Clock drift idnetification and detrending took %s seconds to compile"%(round(time.time() - ts,4)))
    
    
    
    