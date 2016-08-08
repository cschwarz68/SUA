import glob
import os
import pandas as pd
import numpy as np
from scipy import signal
from datetime import timedelta

KPH2MPS = 1/3.6
G2MPSS = 9.8
FS = 10
MINSATELLITES = 4

def read_sub(sub):
    ''' Read all the 10 hz files for one subject
    
    Arguments:
    a subject number is passed into the function as a string like: '001'
    Returns:
    The complete data frame of all a subject's runs is returned
    '''
    print os.getenv('SuaProcessed')
    global categories
    filenames = glob.glob(os.path.join(
        os.getenv('SuaData'), 
        '*' + sub + '*', 
        '*', 
        'data 10hz*.csv'
        ))
    frame = pd.DataFrame()
    dflist = [] 
    
    for filename in filenames:
        categories = []
        print filename
        trip = filename[-8:-4]
        try:
            df = pd.read_csv(filename, 
                usecols=['subject_id', 'time', 'gpstime', 'latitude', 
                    'longitude', 'gpsspeed', 'heading', 'pdop', 'hdop', 'vdop', 
                    'fix_type', 'num_sats', 'acc_x', 'acc_y', 'acc_z', 
                    'throttle', 'obdspeed', 'rpm'],
                parse_dates=[1, 2], 
                infer_datetime_format=True, 
                error_bad_lines=False)
        except Exception:
            print 'read_csv failed on: ' + filename
            continue
            
        # check to see that there are valid gps values and the vehicle is moving
        if missing_gps(df):
            continue
            
        # trim size of file by getting rid of empty rows, duplicates, and null times
        df = trim_file(df)
        df=df.drop_duplicates(subset=['gpstime','latitude','longitude',
            'gpsspeed', 'heading', 'pdop', 'hdop', 'vdop', 'fix_type', 
            'num_sats', 'acc_x', 'acc_y', 'acc_z','throttle', 'obdspeed', 
            'rpm'], keep='first')
        df = df[df.gpstime.notnull()]
        
        # check that the resulting dataframe is not too short now
        if too_short(df):
            continue
        
        #for known problem files, replace the wrong time by gpstime
        numfixed=0
        if trip in open("problem_files.txt").read():
            df, numfixed = replace_time(df,numfixed)
              
        # check that the resulting dataframe is not too short now
        if too_short(df):
            continue

        # combine obd and gps speeds and filter
        # also derive the longitudinal acceleration and add to df
        df = filt_speed(df)
        
        #revise heading to make it smoother add to df
        #also derive the yaw rate and add to df
        df = add_yaw(df)
        
        # pull the trip number from the file name
        df['trip']=df['subject_id'].map(lambda x:trip)   
        
        # drop the raw variables gpsspeed and obdspeed in favor of derived speed
        df = df.drop(['gpsspeed','obdspeed'], 1)  
        
        #find the reverse variable
        df = find_reverse(df,indexLeftSP)
        
        #reformat the column orders
        df=df.reindex(columns=['subject_id', 'time', 'gpstime', 'latitude',
            'longitude', 'heading', 'new_heading', 'yaw_rate', 
            'pdop', 'hdop', 'vdop', 'fix_type', 'num_sats',
            'acc_x', 'acc_y', 'acc_z', 'throttle', 'rpm', 'speed',
            'Ax', 'trip', 'reverse?']) 
        
        # check that the resulting dataframe is not too short now
        if too_short(df):
            continue
           
        dflist.append(df)             
          
    # combine list of frames into one dataframe
    frame = pd.concat(dflist,axis=0)
    
    # export dataframe to csv file
    frame.to_csv(os.path.join(os.getenv('SuaProcessed'), 
        'sub_' + sub + '.csv'), index=None)
    
    #save row count to txt file  
    f = open((os.path.join(os.getenv('SuaProcessed'), "countRows.txt")),'a') 
    f.write('\nsub_' + sub + ', ' + str(len(frame)) + ', ' + str(numfixed))
    f.close()
    return frame   
            
def missing_gps(df):
    ''' Reject a file if there is no gps movement '''
    if not(any(pd.notnull(df.gpsspeed))):
        print "gps no values"
        return True
    if max(df.gpsspeed[pd.notnull(df.gpsspeed)]) == 0:
        print "gps not moving"
        return True
    return False
    
def too_short(df):
    if len(df.gpstime)<=600:
        print "time is less than 60 sec"
        return True
    return False

def trim_file(df):
    ''' Trim the beginning and end of a file based on speed '''
    ismovingG = np.where(df.gpsspeed > 0)[0]
    ismovingO = np.where(df.obdspeed > 0)[0]
    if ismovingO != []:
        if ismovingG[0] > ismovingO[0]:       
            ismoving = df.obdspeed > 0 
        else:
            ismoving = df.gpsspeed > 0
        if ismovingG[-1] > ismovingO[-1]:
            isstopped = df.gpsspeed > 0
        else:
            isstopped = df.obdspeed > 0
    else:
        ismoving = df.gpsspeed > 0
        isstopped = df.gpsspeed > 0
    idx_first = np.where(ismoving)[0][0]
    idx_last = np.where(isstopped)[0][-1]
    try:
        df = df[idx_first:idx_last+1]
    except Exception:
        df = df[idx_first:idx_last]  
    return df   
     
def replace_time(df, numfixed):
    ''' Replace time for any rows that have missing or repeated data '''
    for ind in range(len(df)-1):
        delta1=df.gpstime.iloc[[ind]] - df.time.iloc[[ind]]
        delta2=df.time.iloc[[ind]] - df.gpstime.iloc[[ind]]
        if any(delta1 > timedelta(seconds=1.5)):
            df.time.iloc[[ind]] = df.gpstime.iloc[[ind]]
            numfixed +=1
        if any(delta2 > timedelta(seconds=1.5)):
            df.time.iloc[[ind]] = df.gpstime.iloc[[ind]]
            numfixed +=1 
    return df, numfixed

def filt_speed(df):
    ''' Use obdspeed when the satellites are under 4 '''
    if any(pd.notnull(df.obdspeed)):
        speed = np.where(df['num_sats']>=MINSATELLITES,df.gpsspeed,df.obdspeed)
        for index in range(len(df)):
            if pd.isnull(speed[index]):
                speed[index]=np.array(df.gpsspeed)[index]         
    else:
        speed = df.gpsspeed

    # filter speed
    b,a = signal.butter(2,0.2)
    speedfilt = signal.filtfilt(b,a,speed)
    accel = np.diff(speedfilt * KPH2MPS) * FS / G2MPSS
    accel = np.insert(accel,0,0)
    df['speed'] = speedfilt
    df['Ax'] = accel
    return df     

def add_yaw(df): 
    ''' Add yaw rate based on the adjusted heading '''
    b,a = signal.butter(2,0.2)
    array_heading=np.array(df.heading)
    countzero=0
    index_head = 0 
    
    #find the first heading and backfill to beginning
    for index in range(len(df)):
        if array_heading[index] == 0:
            countzero+=1
        else:
            break
    if len(df) > countzero:
        array_heading[:countzero]=array_heading[countzero] 
        
    #locate jumps and modify               
    if array_heading[0]<180:
        array_heading[0]+=360    
    while index_head < len(array_heading)-1:
        if array_heading[index_head]-array_heading[index_head+1]>350:
            array_heading[index_head+1]+=360
        index_head+=1
    df['new_heading']=pd.Series(array_heading,index=df.index)
    
    #filt new heading and calculate the yaw rate
    filt_heading = signal.filtfilt(b,a,array_heading) 
    yaw = np.diff(filt_heading)
    yaw = np.insert(yaw,0,0)
    df['yaw_rate']=pd.Series(yaw,index=df.index) 
    return df

categories = []
indexLeftSP = 0
def find_reverse(df,indexLeftSP):
    ''' Add reverse column based on heading angle '''
    array_heading = np.array(df.new_heading)    
    for index in range(indexLeftSP, len(array_heading)-1):
        diff = array_heading[index+1]-array_heading[index]
        if len(categories)==len(array_heading)-1:
            break
        if abs(diff)<150 or abs(diff)>210:
            categories.append(0)
        else:
            categories.append(0)
            countR = 1
            for sindex in range(index+1,len(array_heading)-1):
                if abs(array_heading[sindex+1]-array_heading[sindex])<50:
                    categories.append(1)
                    countR+=1
                else:
                    categories.append(1)
                    indexLeftSP = countR + index+1  
                    find_reverse(df, indexLeftSP)                     
                    return df
    categories.append(categories[len(array_heading)-2])
    #Reverse 1 and 0 since forward must be longer than reverse
    if sum(categories)>len(categories)-sum(categories):
        for index, item in enumerate(categories):
            if item == 1:
                categories[index]=0
            else:
                categories[index]=1                                         
    df['reverse?']=categories           
    return df

if __name__ == '__main__':
    import cProfile
    import pstats
    sub = '014'
    cProfile.run('read_sub(sub)', 'nddatastats')
    p = pstats.Stats('nddatastats')
    p.sort_stats('cumulative').print_stats(10)