'''Script Intent: Interface with Kleinschmidt's Proprietary Acoustic Telemetry
Data Management and Positioning Software to create a project database and
import raw data.

Script Author: KPN'''


import os
import sys
import jsats3d
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

outputWS = r"J:\3870\013\Calcs\Production\Data"
inputWS = r"J:\3870\013\Calcs\Production\Data"
dbName = 'cowlitz_2018_electric_boogaloo.db'
dbDir = os.path.join(inputWS,dbName)

# model parameters
utc_conv = -7                                                                   # UTC conversion to local time zone
bm_elev = 861.5/3.28084                                                          # benchmark elevation
bm_elev_units = 'meters'                                                        # why?
masterReceiver = 'R05'                                                          # master receiver, often times the most central 
EPSG = 26910                                                                    # EPSG for UTM Zone 10 NAD 1983
output_units = 'meters'                                                         # we look stupid when we use feet
synch_time = "2018-06-07 13:17:08"                                              # when was the last receiver synchronized - this is zero our, we can't have any detections before  
# create project database
acoustic.create_project_db(outputWS, dbName)
print 'project database created'

# import data to Python
tblTag = pd.read_csv(os.path.join(inputWS,'tblTag.csv'))
tblReceiver = pd.read_csv(os.path.join(inputWS,'tblReceiver_initial_B.csv'), dtype = {'Rec_ID':str,'Type':str,'Tag_ID':str,'Ref_Elev':str,'X':np.float64,'Y':np.float64,'Z':np.float64,'X_t':np.float64,'Y_t':np.float64,'Z_t':np.float64,})
tblWSEL = pd.read_csv(os.path.join(inputWS,'tblWSEL.csv'))
tblTemp = pd.read_csv(os.path.join(inputWS,'tblTemp.csv'))

# write data to SQLite
acoustic.study_data_import(tblTag,dbDir,'tblTag')
print 'tblTag imported'
acoustic.study_data_import(tblReceiver,dbDir,'tblReceiver')
print 'tblReceiver imported'
acoustic.study_data_import(tblWSEL,dbDir,'tblWSEL')
print 'tblWSEL imported'
acoustic.study_data_import(tblTemp,dbDir,'tblTemp')
print 'tblTemp imported'

# write study parameters to database
acoustic.set_study_parameters(utc_conv,bm_elev,bm_elev_units,output_units,masterReceiver,synch_time,dbDir)

# import raw data files per receiver
#R01
r01Dat = r"C:\Users\Kevin Nebiolo\Desktop\Cowlitz_Desktop_091318\Data\Aggregated\R01"
acoustic.acoustic_data_import('R01','Teknologic',r01Dat,dbDir)
#R02
r02Dat = r"C:\Users\Kevin Nebiolo\Desktop\Cowlitz_Desktop_091318\Data\Aggregated\R02"
acoustic.acoustic_data_import('R02','Teknologic',r02Dat,dbDir)
#R03
r03Dat = r"C:\Users\Kevin Nebiolo\Desktop\Cowlitz_Desktop_091318\Data\Aggregated\R03"
acoustic.acoustic_data_import('R03','Teknologic',r03Dat,dbDir)
#R04
r04Dat = r"C:\Users\Kevin Nebiolo\Desktop\Cowlitz_Desktop_091318\Data\Aggregated\R04"
acoustic.acoustic_data_import('R04','Teknologic',r04Dat,dbDir)
#R05
r05Dat = r"C:\Users\Kevin Nebiolo\Desktop\Cowlitz_Desktop_091318\Data\Aggregated\R05"
acoustic.acoustic_data_import('R05','Teknologic',r05Dat,dbDir)
#R06
r06Dat = r"C:\Users\Kevin Nebiolo\Desktop\Cowlitz_Desktop_091318\Data\Aggregated\R06"
acoustic.acoustic_data_import('R06','Teknologic',r06Dat,dbDir)
#R07
r07Dat = r"C:\Users\Kevin Nebiolo\Desktop\Cowlitz_Desktop_091318\Data\Aggregated\R07"
acoustic.acoustic_data_import('R07','Teknologic',r07Dat,dbDir)
#R08
r08Dat = r"C:\Users\Kevin Nebiolo\Desktop\Cowlitz_Desktop_091318\Data\Aggregated\R08"
acoustic.acoustic_data_import('R08','Teknologic',r08Dat,dbDir)
#R09
r09Dat = r"C:\Users\Kevin Nebiolo\Desktop\Cowlitz_Desktop_091318\Data\Aggregated\R09"
acoustic.acoustic_data_import('R09','Teknologic',r09Dat,dbDir)

print 'Project Database Set Up - Proceed to Detection import and initial filtering'

