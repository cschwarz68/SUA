import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_df(tripNum):
    '''return df of the trip in the subject file.'''
    
    #find the sub number of the trip
    f=open(os.path.join(os.getenv('SuaProcessed'),"trip_info.txt"))
    tripInfo = f.readlines()
    for row in tripInfo:
        if str(tripNum) in row:
            subNum = row[:7]
            print 'trip: ', row
            break
            
    #find df of the trip  
    subname = os.path.join(os.getenv('SuaProcessed'), subNum+'.csv')      
    sub_df = pd.read_csv(subname)   
    df = sub_df[sub_df.trip == tripNum]
    return df

def graph_lat_lng(tripNum):
    '''plot longitude and latitude of the trip with reverse in red'''
    
    df=find_df(tripNum)
    if len(df) == 0:
        print 'the trip did not move or was less than 60 sec'
        return 
    categories=df['reverse?']
    plt.figure()      
    colormap = np.array(['0.85', 'r'])
    plt.scatter(np.array(df.longitude), np.array(df.latitude), c=colormap[categories],linewidths=0)
    plt.plot(np.array(df.longitude), np.array(df.latitude),'0.85')
    plt.grid(True)
    plt.show()
    
    isReversing = np.where(df['reverse?'] == 1)[0]
    if len(isReversing) >1:
        print 'Reverse Loc: ',(float(df.latitude.iloc[[isReversing[0]]]), (float(df.longitude.iloc[[isReversing[0]]])))
        for idn in range(1,len(isReversing)):       
            if isReversing[idn]-isReversing[idn-1] !=1:                      
                print 'Reverse Loc: ',(float(df.latitude.iloc[[isReversing[idn]]]), float(df.longitude.iloc[[isReversing[idn]]]))
    else:
        print "No reverse in this trip"

graph_lat_lng(1894)  

