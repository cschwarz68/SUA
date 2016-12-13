import glob
import os
import pandas as pd

def getFileInfo(filename,column_name):
    '''Put all subjects and trips into a txt file.'''
    tolTripNum = 0
    
    d=os.path.join(os.getenv('SuaData'),"Driver_Event Query_Verififed_NADS.csv")
    DR_df = pd.read_csv(d) 
    DR_df = DR_df[['Run ID',column_name]] 
    
    for subject in glob.glob(os.path.join(os.getenv('SuaData'),"*")):
        print 'subjectfile: ', subject[17:20]     
        trips = glob.glob(subject+"\* run*")
        tolTripNum += len(trips)
        for trip in trips:
            if trip[-4:] != 'xlsx':
                f=open((os.path.join(os.getenv('SuaProcessed'),filename+".txt")),'a')                
                f.write('\nsub_' + subject[17:20] + ', ' + trip[-4:])
                for index in range(len(DR_df)):
                    if str(trip[-4:]) in str(DR_df['Run ID'][index]):
                        f.write(', ' + str(DR_df[column_name][index]))
                    elif str(trip[-4])==' ' and str(trip[-3:]) == str(DR_df['Run ID'][index]):
                        f.write(', ' + str(DR_df[column_name][index]))  
                    elif str(trip[-4])==' ' and str(trip[-3:])+'a' == str(DR_df['Run ID'][index]):
                        f.write(', ' + str(DR_df[column_name][index])) 
                    elif str(trip[-4])==' ' and str(trip[-3:])+'b' == str(DR_df['Run ID'][index]):
                        f.write(', ' + str(DR_df[column_name][index]))                       
                f.write('                  ')                                                                                     
                f.close()
    print tolTripNum
    
getFileInfo('trip_info_beg','Initial gear selection') 
getFileInfo('trip_info_end','Maneuver type end')
