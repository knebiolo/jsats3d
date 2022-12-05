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

outputWS = r"C:\Users\knebiolo\Desktop\jsats with DBSCAN\Output\Scratch"
figureWS = r"C:\Users\knebiolo\Desktop\jsats with DBSCAN\Output\Figures"
inputWS = r"C:\Users\knebiolo\Desktop\jsats with DBSCAN\Output"
dbName = 'cowlitz_test2.db'
dbDir = os.path.join(inputWS,dbName)
#dbDir = r"C:\Users\Kevin Nebiolo\Desktop\cowlitz_tagDrag.db"

# get list of tags
conn = sqlite3.connect(dbDir)
c = conn.cursor()
# tags = pd.read_sql('SELECT Tag_ID FROM tblTag WHERE Species == "Steelhead"',con = conn).Tag_ID.values
# tags = np.sort(tags)    
tags = ['022E']
#tags = ['01C6','022E','0A88']
      
c.close()
print ("There are %s fish to iterate through"%(len(tags)))

# for i in tags:
#     obj = jsats3d.multipath_data_object(i,dbDir,outputWS,metronome = True)                 # for every tag, make a multipath data object 
#     jsats3d.multipath_2(obj)                                                        # apply primary filter
#     print("Primary filter applied to tag %s"%(i))

#jsats3d.multipath_data_management(outputWS,dbDir,primary = True)          # add data to database

# apply secondary filter
for i in tags:
    print ("Secondary mulitpath filter applied to tag %s"%(i))
    #jsats3d.multipath_classifier(i,dbDir,outputWS,beacon = True, metronome = True, method = "KNN")          # apply secondary filter
    jsats3d.multipath_classifier(i,dbDir,outputWS,beacon = False,method = "KNN")          # apply secondary filter

#jsats3d.multipath_data_management(outputWS,dbDir,primary = False)         # add data to database

print ("All tags processed, proceed to positioning")


    
