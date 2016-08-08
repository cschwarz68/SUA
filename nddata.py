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
not_match_number = 0

def read_sub(sub):
    ''' Read all the 10 hz files for one subject
    
    Arguments:
    a subject number is passed into the function as a string like: '001'
    Returns:
    The complete data frame of all a subject's runs is returned
    '''        
    global categories
    global not_match_number
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
                        'throttle', 'rpm'],
                parse_dates=[1, 2], 
                infer_datetime_format=True, 
                error_bad_lines=False)
        except Exception:
            print 'read_csv failed on: ' + filename
            continue
                
        # check to see that there are valid gps values and the vehicle is moving   
        if missing_gps(df):
            continue    
            
        #find the difference between acc rows and speed rows
        if any(pd.notnull(df.acc_x)):
            hasacc,hasacc1 = df.acc_x > 0,df.acc_x<0
            if len(np.where(hasacc1)[0]) > 0:
                np.where(hasacc1)[0][0] 
                idx_acc,idx_acc1 = np.where(hasacc)[0][0],np.where(hasacc1)[0][0]
                idx_first_acc = min(idx_acc,idx_acc1) 
            else:
                idx_first_acc = np.where(hasacc)[0][0]
            
            ismoving = df.gpsspeed > 0
            idx_moving = np.where(ismoving)[0][0]
            acc_diff = idx_moving - idx_first_acc   
        else:
            acc_diff = 0
                                                                                    
        #trim size of file by getting rid of empty rows, duplicates, and null times
        df = trim_file(df)
        df=df.drop_duplicates(subset=['gpstime','latitude','longitude',
            'gpsspeed', 'heading', 'pdop', 'hdop', 'vdop','fix_type', 'num_sats', 
            'acc_x', 'acc_y', 'acc_z','throttle', 'rpm'],
            keep='first')
        df = df[df.gpstime.notnull()]
        
        #Staff driving, which should be deleted
        if (trip =='2968' or trip =='7804'):
            df = df[0:0]
            
        #check that the resulting dataframe is not too short now
        if too_short(df):
            continue
                    
        #for known problem files, replace the wrong time by gpstime
        if trip in open("problem_files.txt").read():
<<<<<<< HEAD
            df, numfixed = replace_time(df,numfixed)
            print('problem file. numfixed = ' + str(numfixed))
              
        # check that the resulting dataframe is not too short now
        if too_short(df):
            continue

        # combine obd and gps speeds and filter
        # also derive the longitudinal acceleration and add to df
=======
            df = replace_time(df)
                     
        #derive the longitudinal acceleration and add to df
>>>>>>> 2f9520ba9345f57692e80bae5a43d9ce76f382c5
        df = filt_speed(df)
                    
        #revise heading to make it smoother and add to df
        #also derive the yaw rate and add to df
        df = add_yaw(df)
                    
        # pull the trip number from the file name
        df['trip']=df['subject_id'].map(lambda x:trip)  
        
        #delete junk rows in the begining of the trips    
        if trip == '2247':
            df = df[7:]
        elif trip == '2462':
            df = df[13:]
                    
        #add reverse variable to df
        #also add manuever status to df
        df = decide_start_status(df, trip, acc_diff)
        df = add_end_status(df, trip)
            
        #reformat the column orders               
        df=df.reindex(columns=['subject_id', 'time', 'gpstime', 'latitude',
            'longitude', 'heading', 'new_heading', 'yaw_rate',
            'pdop', 'hdop', 'vdop', 'fix_type', 'num_sats',
            'acc_x', 'acc_y', 'acc_z', 'throttle', 'rpm', 'speed',
            'Ax', 'trip', 'reverse?', 'manuev_init', 'manuev_end']) 
    
        #if the begining speed is too big,then df misses starting gps
        df = big_starting_spd(df, sub, trip)   
         
        #check if the speed of reversing period is too high  
        df = big_reversing_spd(df)
        
        #if the reversing period is too long, then change it to forward                       
        L1,L2 = create_list(df,l1=[],l2=[],count=0,List1=[],List2=[]) 
        if trip in open("change_rvs2fwd_after_videos.txt").read():                   
            for index in range(len(L1)):
                count_zero = L2[index].count(0)
                num_rev = len(L2[index])-count_zero
                if (L1[index][0] == 1 and (num_rev > 200)):  
                    L1[index] = [0]*len(L1[index])   
            new_rvs = sum(L1,[])
            df['reverse?'] = new_rvs   
            
        #revise the manuev_init status    
        if np.array(df['reverse?'])[0] == 0:          
            df['manuev_init'] = 'D'
        else:
            df['manuev_init'] = 'R'
        
        #check if the end status match        
        check = np.array(df['reverse?'])[-1]
        trip_end = np.array(df.manuev_end)[0]
        if (check ==0 and trip_end == 'R') or (check == 1 and trip_end == 'D'):
            print ' '
            print 'end status not match'
            print ' '
            not_match_number +=1
            f = open((os.path.join(os.getenv('SuaProcessed'), "end_status_not_match.txt")),'a') 
            f.write('\nsub_' + sub + ', '+ trip + ', ' +str(check)+', '+trip_end+', '+str(not_match_number) )
            f.close()                                                                                              
                            
        dflist.append(df)             
                    
    #combine list of frames into one dataframe
    frame = pd.concat(dflist,axis=0)
     
    #export dataframe to csv file
    frame.to_csv(os.path.join(os.getenv('SuaProcessed'), 
        'sub_' + sub + '.csv'), index=None)
                                                      
    #save row count and number of row-fixed to txt file  
    f = open((os.path.join(os.getenv('SuaProcessed'), "countRows.txt")),'a') 
    f.write('\nsub_' + sub + ', ' + str(len(frame)))
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
    ismoving = df.gpsspeed > 0
    idx_first = np.where(ismoving)[0][0]
    idx_last = np.where(ismoving)[0][-1]
    try:
        df = df[idx_first:idx_last+1]
    except Exception:
        df = df[idx_first:idx_last]  
    return df   
     
def replace_time(df):
    ''' Replace time for any rows that have missing or repeated data '''
    for ind in range(len(df)-1):
        delta1=df.gpstime.iloc[[ind]] - df.time.iloc[[ind]]
        delta2=df.time.iloc[[ind]] - df.gpstime.iloc[[ind]]
        if any(delta1 > timedelta(seconds=1.5)):
            df.time.iloc[[ind]] = df.gpstime.iloc[[ind]]

        if any(delta2 > timedelta(seconds=1.5)):
            df.time.iloc[[ind]] = df.gpstime.iloc[[ind]]
    return df

def filt_speed(df):
    '''derive the longitudinal acceleration and add to df'''
    speed = df.gpsspeed
    b,a = signal.butter(2,0.2)
    if len(speed) >= 3*max(len(a),len(b)):
        speedfilt = signal.filtfilt(b,a,speed)
    else:
        speedfilt = speed
    accel = np.diff(speedfilt * KPH2MPS) * FS / G2MPSS
    accel = np.insert(accel,0,0)
    df['speed'] = speed
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

def decide_start_status(df,trip,acc_diff):
    ''' Get the reverse info based on the starting status and add manuev_init to df'''
    f = open((os.path.join(os.getenv('SuaProcessed'),"trip_info_beg.txt")))       
    read = f.readlines()
    for row in read:
        if trip in row:
            trip_start = row[15]
            break
    f.close()
    
    if trip_start == 'D':
        df = start_in_drive(df,indexLeftSP = 0, categories = [])
    elif acc_diff > 600:
        df = start_in_drive(df,indexLeftSP = 0, categories = [])
    elif trip_start == 'R':
        if trip == '2247':
            df = start_in_reverse(df,indexLeftSP = 0, categories = [], lowbd=90,highbd=270)
        else:
            df = start_in_reverse(df,indexLeftSP = 0, categories = [], lowbd=65,highbd=295)
    else:
        trip_start = ' '
        df = start_in_drive(df,indexLeftSP = 0, categories = [])
        if sum(df['reverse?'])> len(df)-sum(df['reverse?']):
            df = start_in_reverse(df,indexLeftSP = 0, categories = [], lowbd=65,highbd=295)
    df['manuev_init'] = trip_start
    return df
   
def add_end_status(df,trip):
    ''' add manuev_end to df based onthe reverse info'''
    f = open((os.path.join(os.getenv('SuaProcessed'),"trip_info_end.txt")))       
    read = f.readlines()
    for row in read:
        if trip in row:
            if 'forward' in row[15:31]:
                trip_end = 'D'
            elif 'Parallel' in row [15:31]:
                trip_end = 'P'
            elif 'Backing' in row[15:31]:
                trip_end = 'R'
            else:
                trip_end = ' '
            break
    f.close() 
    df['manuev_end'] = trip_end  
    return df

def start_in_drive(df,indexLeftSP,categories):
    ''' Add reverse column based on heading angle when starting in Forward'''
    array_heading = np.array(df.new_heading)   
        
    for index in range(indexLeftSP, len(array_heading)-1):
        diff1 = array_heading[index+1]-array_heading[index]
        if len(categories)==len(array_heading)-1:
            break
        if abs(diff1)%360 < 150 or abs(diff1)%360 > 210:
            categories.append(0)
        else:
            categories.append(0)
            countR = 1
            for sindex in range(index+1,len(array_heading)-1):
                diff2 = array_heading[sindex+1]-array_heading[sindex]
                if abs(diff2)%360 < 65 or abs(diff2)%360 > 295:
                    categories.append(1)
                    countR+=1                  
                else:
                    categories.append(1)
                    indexLeftSP = countR + index+1  
                    start_in_drive(df, indexLeftSP,categories)                     
                    return df
    categories.append(categories[len(array_heading)-2])
    df['reverse?']=categories                                  
    return df

def start_in_reverse(df,indexLeftSP,categories,lowbd,highbd):
    ''' Add reverse column based on heading angle when starting in Reverse'''
    countF=0
    array_heading = np.array(df.new_heading)
    for index in range(indexLeftSP, len(array_heading)-1):
        diff1 = array_heading[index+1]-array_heading[index]
        if len(categories)==len(array_heading)-1:
            break
        if abs(diff1)%360 <= lowbd or abs(diff1)%360 >= highbd:
            categories.append(1)
        else:
            categories.append(1)
            for sindex in range(index+1,len(array_heading)-1):
                diff2 = array_heading[sindex+1]-array_heading[sindex]
                if abs(diff2)%360 < 150 or abs(diff2)%360 > 210:
                    categories.append(0)
                    countF+=1
                else:
                    categories.append(0)
                    indexLeftSP = countF+index+2
                    start_in_reverse(df,indexLeftSP,categories,lowbd,highbd)
                    return df
    categories.append(categories[len(array_heading)-2])
    df['reverse?'] = categories
    return df

def big_starting_spd(df,sub,trip):
    '''check if the begining speed is higher than 16.If it is, then start driving by forward'''
    reverse = np.array(df['reverse?'])
    speed = np.array(df['speed'])
    if reverse[0] == 1:
        for index in range(len(reverse)):
            if reverse[index] == 1:
                if speed[index] > 16:
                    start_in_drive(df,indexLeftSP=0,categories=[])
                    break
            else:
                break      
    return df

#List1 for reverse variable
#List2 for speed variable 
List1 = []
List2 = []
def create_list(df,l1,l2,count,List1,List2):    
    '''create two lists of lists for reverse and speed variables'''
    speed = np.array(df.speed)
    reverse = np.array(df['reverse?'])
    for index in range(count,len(reverse)):
        if index == len(reverse)-1:
            l1.append(reverse[-1])
            l2.append(speed[-1])
            List1.append(l1)
            List2.append(l2)
            break
        elif reverse[index] == reverse[index+1]:         
            l1.append(reverse[index])
            l2.append(speed[index])
        else:
            l1.append(reverse[index])
            l2.append(speed[index])
            count = count+len(l1)
            List1.append(l1)
            List2.append(l2)
            l1 = []
            l2 = []
            create_list(df,l1,l2, count,List1,List2)
            return List1,List2
    return List1,List2

def big_reversing_spd(df):
    '''revise the reversing speed higher than 16 to forward'''
    List1, List2 = create_list(df,l1=[],l2=[],count=0,List1=[],List2=[])
    final_list = []
    for index in range(len(List2)):
        if max(List2[index]) > 16 and 1 in List1[index]:
            final_list.extend([0]*len(List1[index]))
        else:
            final_list.extend(List1[index])    
    df['reverse?'] = final_list
    return df
           
if __name__ == '__main__':
    import cProfile
    import pstats
<<<<<<< HEAD
    sub = '058'
=======
    sub = '001'
>>>>>>> 2f9520ba9345f57692e80bae5a43d9ce76f382c5
    cProfile.run('read_sub(sub)', 'nddatastats')
    p = pstats.Stats('nddatastats')
    p.sort_stats('cumulative').print_stats(10)
    
