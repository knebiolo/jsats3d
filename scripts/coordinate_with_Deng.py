'''Script Intent: Interface with Kleinschmidt's Proprietary Acoustic Telemetry
Data Management and Positioning Software to coordinate fish using Deng's exact 
solution.

Script utilizes multiprocessing, make sure the number of processes is not greater
than the number of cores you have on your computer.

Script Author: KPN'''

# Import Modules:
import os
import jsats3d
import warnings
warnings.filterwarnings('ignore')
import numpy as np

print ("Modules Imported, set parameters")
outputWS = r"J:\3870\013\Calcs\Paper\Output\Scratch"
figureWS = r"J:\3870\013\Calcs\Paper\Output\Figures"
inputWS = r"J:\3870\013\Calcs\Paper\Data"
dbName = 'cowlitz_2018_paper.db'
dbDir = os.path.join(inputWS,dbName)
#recList = ['R01','R02','R03','R05','R06','R07','R08','R09']
recList = ['R01','R02','R03','R05','R06','R07','R08','R09']

tags = ['01C6','022E','0A88','21C2']
  
testTag = '022E'

# create a position object for our test tag
pos = jsats3d.position(testTag,recList,dbDir,outputWS,figureWS)
print ("position object created, initialize Deng's solution")
# coordinate using Deng's exact method
pos.Deng()
pos.trajectory_plot_Deng(rolling_avg_window = 8)

# # when positioning receivers - calculate median X, Y, and Z
# solA = pos.DengSolutionA_unfiltered[pos.DengSolutionA_unfiltered.comment == 'solution found']
# solB = pos.DengSolutionB_unfiltered[pos.DengSolutionB_unfiltered.comment == 'solution found']

# # A solutions
# Xa = solA.X.median()
# Ya = solA.Y.median()
# Za = solA.Z.median()

# # B solutions
# Xb = solB.X.median()
# Yb = solB.Y.median()
# Zb = solB.Z.median()

# # print
# print ("A solutions produced position of %s at %s,%s,%s"%(testTag,np.round(Xa,2),np.round(Ya,2),np.round(Za,2)))
# print ("B solutions produced position of %s at %s,%s,%s"%(testTag,np.round(Xb,2),np.round(Yb,2),np.round(Zb,2)))
