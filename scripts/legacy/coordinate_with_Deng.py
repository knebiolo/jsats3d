'''Script Intent: Position with Deng (2011)

Script Author: KPN'''

# Import Modules:
import os
import jsats3d
import warnings
warnings.filterwarnings('ignore')
import numpy as np

print ("Modules Imported, set parameters")
outputWS = r"C:\Users\knebiolo\Desktop\jsats with DBSCAN\Output"
figureWS = r"C:\Users\knebiolo\Desktop\jsats with DBSCAN\Output\Figures"
inputWS = r"C:\Users\knebiolo\Desktop\jsats with DBSCAN\Output"
dbName = 'cowlitz_test2.db'
dbDir = os.path.join(inputWS,dbName)

recList = ['R01','R02','R04','R03','R05','R06','R07','R08','R09']
#recList = ['R04','R05','R06''R07','R08','R09']

  
testTag = '21C2'

# create a position object for our test tag
pos = jsats3d.position(testTag,recList,dbDir,outputWS,figureWS)
print ("position object created, initialize Deng's solution")
# coordinate using Deng's exact method
pos.Deng(print_output = True)
#pos.trajectory_plot_Deng(beacon = True)
pos.trajectory_plot_Deng()

