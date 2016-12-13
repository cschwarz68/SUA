import glob
import os
import pandas as pd
import numpy as np
from scipy import signal, interpolate
from datetime import timedelta
import matplotlib.pyplot as plt

KPH2MPS = 1/3.6
G2MPSS = 9.8
FS = 10
MINSATELLITES = 4
MAKEPLOTS = True
WRITEFILE = True
WRAPYAWANGLE = 290
not_match_number = 0

def read_sub(sub,single_trip=0):
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

        # allow for a single trip number to be passed in as an optional arg
        if single_trip > 0 and trip != single_trip:
            continue
                
        # for known files with unfixable problems, skip them
        if trip in open("skip_files.csv").read():
            continue
                                        
        # check to see that there are valid gps values and the vehicle is moving   
        if missing_gps(df):
            continue    
            
        # find the number of rows that acc_x has values before gpsspeed starts
        idx_moving, idx_acc, idx_change = key_indices(df)
        if (idx_moving is np.nan) or (idx_acc is np.nan) or (idx_change is np.nan):
            continue
        acc_diff = idx_change - idx_acc
                                                                                            
        # trim size of file by getting rid of empty rows, duplicates, and null times
        df = trim_file(df,idx_moving,idx_acc,idx_change)
        df = df.drop_duplicates(subset=['gpstime','latitude','longitude','gpsspeed'],
            keep='first')
        df = df[df.gpstime.notnull()]
                    
        # check that the resulting dataframe is not too short now
        if too_short(df):
            continue

        #df = df.set_index(df.gpstime)
                    
        # for known problem files, replace the wrong time by gpstime
        if trip in open("problem_files.txt").read():
            df = replace_time(df)
   
        # combine obd and gps speeds and filter
        # also derive the longitudinal acceleration and add to df
        df = filt_speed(df)
        
        # revise heading to make it smoother and add to df
        # also derive the yaw rate and add to df
        df = add_yawrate(df)

        df = df.reset_index(drop=True)
        
        # diagnostic plots 
        if MAKEPLOTS:    
            F = plt.figure()
            plt.subplot(321)
            plt.plot(df.gpstime)
            plt.title(trip)
            plt.subplot(322)
            plt.plot(df.Ax)
            plt.hold
            plt.plot(-df.acc_x1)
            plt.ylim(-0.3,0.3)
            plt.ylabel('Ax (G)')
            plt.subplot(323)
            plt.plot(df.gpsspeed)
            plt.hold
            plt.plot(df.speed)
            a = np.array(plt.axis())
            a[0] = df.index[0]
            plt.axis(a)
            plt.ylabel('speed (kph)')
            plt.subplot(324)
            plt.plot(df.yaw_rate)
            plt.ylim(-4,4)
            plt.ylabel('deg/s')
            plt.subplot(325)
            plt.plot(df.new_heading)
            plt.ylabel('yaw (deg)')       
            plt.subplot(326)
            plt.plot(df.acc_y1)
            plt.ylim(-0.3,0.3)
            plt.ylabel('Ay (G)')
            F.set_size_inches(16, 6)
            #plt.pause(1)
            plt.savefig(os.path.join('plots',sub + '_' + trip + '.png'))
                                           
        # pull the trip number from the file name
        df['trip']=df['subject_id'].map(lambda x:trip)  
        
        # delete junk rows in the begining of the trips    
        if trip == '2247':
            df = df[7:]
        elif trip == '2462':
            df = df[13:]
                    
        # add reverse variable to df
        # also add manuever status to df
        df = decide_start_status(df, trip, acc_diff)
        df = add_end_status(df, trip)
            
        # reformat the column orders               
        df=df.reindex(columns=['subject_id', 'time', 'gpstime', 'latitude',
            'longitude', 'heading', 'new_heading', 'yaw_rate',
            'pdop', 'hdop', 'vdop', 'fix_type', 'num_sats',
            'acc_x', 'acc_y', 'acc_z', 'throttle', 'rpm', 'speed',
            'Ax', 'trip', 'reverse?', 'manuev_init', 'manuev_end']) 
    
        # if the begining speed is too big,then we think the df misses starting gps
        df = big_starting_spd(df, sub, trip)   
         
        # check if the speed of reversing period is too high  
        df = big_reversing_spd(df)
        
        # if the reversing period is too long, then change it to forward                       
        L1,L2 = create_list(df,l1=[],l2=[],count=0,List1=[],List2=[]) 
        if trip in open("change_rvs2fwd_after_videos.txt").read():                   
            for index in range(len(L1)):
                count_zero = L2[index].count(0)
                num_rev = len(L2[index])-count_zero
                if (L1[index][0] == 1 and (num_rev > 200)):  
                    L1[index] = [0]*len(L1[index])   
            new_rvs = sum(L1,[])
            df['reverse?'] = new_rvs   
            
        # revise the manuev_init status    
        if np.array(df['reverse?'])[0] == 0:          
            df['manuev_init'] = 'D'
        else:
            df['manuev_init'] = 'R'
        
        # check if the end status match        
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
                    
    # combine list of frames into one dataframe
    frame = pd.concat(dflist,axis=0)
                                              
    # save row count and number of row-fixed to txt file  
    if WRITEFILE:
        # export dataframe to csv file
        frame.to_csv(os.path.join(os.getenv('SuaProcessed'), 
        'sub_' + sub + '.csv'), index=None)
        # export countRows data to file
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
    ''' Reject a file if it doesn't cover enough time '''
    if len(df.gpstime)<=600:
        print "time is less than 60 sec"
        return True
    return False            

def key_indices(df):
    ''' 
    find some key indices that indicate when good data begins 
    idx_moving is the index of the first frame the speed is greater than 0.
    idx_acc is the index of the first frame that Ax is numeric.
    idx_change is the index of the first frame the speed is consistently changing.
    There are periods of bad speed where the gpsspeed could be non-zero, but
    not changing. It is still bad until it starts varying appropriately
    '''
    ismoving = df.gpsspeed > 0
    idx_moving = np.where(ismoving)[0]
    if idx_moving.size>0:
        idx_moving = idx_moving[0]
    else:
        idx_moving = np.nan
    
    has_acc = pd.notnull(df.acc_x)
    idx_acc = np.where(has_acc)[0]
    if idx_acc.size>0:
        idx_acc = idx_acc[0]
    else:
        idx_acc = np.nan
    
    speed_diff = np.diff(df.gpsspeed)
    speed_diff = np.insert(speed_diff,0,speed_diff[0])
    isvalid = np.logical_and(abs(speed_diff)>0.00001,abs(speed_diff)<1.0)
    change = pd.rolling_sum(isvalid,10)
    idx_changed = np.where(change>=5)[0]
    if idx_changed.size>0:
        idx_change = max(0,idx_changed[0]-5)
    else:
        idx_change = np.nan
    
    return idx_moving, idx_acc, idx_change

def trim_file(df,idx_moving,idx_acc,idx_change):
    ''' 
    Trim the beginning and end of a file.
    The file should begin when there is gps speed signal and
    end when the speed has dropped to zero
    '''
    #df = df[df.gpstime.notnull()]
    df.loc[0:idx_change,'gpsspeed'] = np.nan
    df.loc[0:idx_change,'heading'] = np.nan
    
    ismoving = df.gpsspeed > 0
    idx_last = np.where(ismoving)[0][-1]
    df = df[max(idx_acc,idx_change):idx_last]
        
    return df   
     
def replace_time(df):
    ''' Replace time for any rows that have missing or repeated data '''
    delta=df.gpstime - df.time
    isover = delta>timedelta(seconds=1.5)
    isunder = delta<timedelta(seconds=-1.5)
    df.loc[isover,'time'] = df.loc[isover,'gpstime']
    df.loc[isunder,'time'] = df.loc[isunder,'gpstime']
    return df

def filt_speed(df):
    ''' derive the longitudinal acceleration and add to df '''
    speed = np.array(df.gpsspeed)
    # interpolate across any missing values
    i_samples = np.array(range(len(speed)))
    isvalid = pd.notnull(speed)
    f = interpolate.interp1d(i_samples[isvalid],speed[isvalid],
        fill_value = 'extrapolate')
    speed = f(i_samples)
    # lower limit the speed at 0
    speed[speed<0.0] = 0.0
    # smooth the speed
    b,a = signal.butter(2,0.2)
    speedfilt = filter_segments(b,a,speed)
    df['speed'] = speedfilt
    # estimate the acceleration with differentiation
    accel = np.diff(speedfilt * KPH2MPS) * FS / G2MPSS
    accel = np.insert(accel,0,accel[0])
    isjump = abs(accel)>0.7
    accel[isjump] = np.nan
    i_samples = np.array(range(len(accel)))
    isvalid = pd.notnull(accel)
    f = interpolate.interp1d(i_samples[isvalid],accel[isvalid],
        fill_value = 'extrapolate')
    accel = f(i_samples)
    df['Ax'] = accel
    # if there are any nans in the acceleration, have to get rid of them
    if all(df.acc_x.isnull()):
        df['acc_x1'] = df.acc_x
    elif any(df.acc_x.isnull()):
        df['i'] = range(len(df))
        isvalid = df.acc_x.notnull()
        f = interpolate.interp1d(df.loc[isvalid,'i'],df.loc[isvalid,'acc_x'],
            fill_value = 'extrapolate')
        acc_x = f(df.i)
        df['acc_x1'] = filter_segments(b,a,acc_x)
    else:
        df['acc_x1'] = filter_segments(b,a,df.acc_x)
    if all(df.acc_y.isnull()):
        df['acc_y1'] = df.acc_y
    elif any(df.acc_y.isnull()):
        df['i'] = range(len(df))
        isvalid = df.acc_y.notnull()
        f = interpolate.interp1d(df.loc[isvalid,'i'],df.loc[isvalid,'acc_y'],
            fill_value = 'extrapolate')
        acc_y = f(df.i)
        df['acc_y1'] = filter_segments(b,a,acc_y)
    else:
        df['acc_y1'] = filter_segments(b,a,df.acc_y)
    return df     

def add_yawrate(df): 
    ''' Add yaw rate based on the adjusted heading '''
    array_heading=np.array(df.heading)
    # derivative of heading
    yaw_rate = np.diff(array_heading)
    yaw_rate = np.insert(yaw_rate,0,yaw_rate[0])
    
    #find the first stable heading and backfill to beginning
    for index in range(len(df)):
        if sum(abs(yaw_rate[index:index+5]))>10:
            continue
        elif ~np.isfinite(sum(abs(yaw_rate[index:index+5]))):
            continue
        else:
            array_heading[:index]=array_heading[index]
            break                
            
    #locate jumps and modify
    while any(abs(yaw_rate)>=WRAPYAWANGLE):
        idx_jump_le, idx_jump_te = find_edges(abs(yaw_rate)>=WRAPYAWANGLE)
        for idx in idx_jump_le:
            if yaw_rate[idx]>0:
                array_heading[idx:] -= 360
            else:
                array_heading[idx:] += 360
        yaw_rate = np.diff(array_heading)
        yaw_rate = np.insert(yaw_rate,0,yaw_rate[0])
    
    df['new_heading']=pd.Series(array_heading,index=df.index)

    # recalculate the yaw rate
    yaw_rate = np.diff(array_heading)
    yaw_rate = np.insert(yaw_rate,0,yaw_rate[0])
    yaw_rate_sign = np.sign(yaw_rate)
    
    # identify large jumps
    isjump = abs(yaw_rate)>10
    yaw_rate[isjump] = np.nan
    i_samples = np.array(range(len(yaw_rate)))
    isvalid = pd.notnull(yaw_rate)
    f = interpolate.interp1d(i_samples[isvalid],yaw_rate[isvalid],
        fill_value = 'extrapolate')
    yaw_rate = f(i_samples)
    b,a = signal.butter(2,0.2)
    yaw_rate2 = filter(b,a,yaw_rate)*FS
    df['yaw_rate'] = pd.Series(yaw_rate2,index=df.index)
        
    return df

def filter_segments(b,a,x):
    x = np.array(x)
    idx_valid_le, idx_valid_te = find_edges(~np.isnan(x))
    for i in range(len(idx_valid_le)):
        x[idx_valid_le[i]:idx_valid_te[i]] = filter(b,a,x[idx_valid_le[i]:idx_valid_te[i]])
    return x
    
def filter(b,a,x):
    if len(x) <= 3*max(len(a),len(b)):
        y = x
        return y
    
    y = signal.filtfilt(b,a,x)
    return y

def find_edges(x):
    ''' find the indices of all the leading and trailing edges of a bool
        array '''
    shift = x
    mask = np.ones(len(shift), dtype=bool)
    mask[-1] = 0
    shift = shift[mask]
    shift = np.insert(shift,0,shift[0])
    
    le = np.logical_and(x==1,shift==0)
    te = np.logical_and(x==0,shift==1)
    idx_le = np.where(le)[0]
    idx_te = np.where(te)[0]
    
    if np.logical_and(idx_le.size==0,idx_te.size==0):
        if any(x):
            idx_le = np.array([1])
            idx_te = np.array([len(x)-1])
        else:
            idx_le = np.array([])
            idx_te = np.array([])
    elif np.logical_and(idx_le.size==0,idx_te.size>0):
        idx_le = np.array([1])
    elif np.logical_and(idx_le.size>0,idx_te.size==0):
        idx_te = np.array([len(x)-1])
    else:
        if idx_le[0] > idx_te[0]:
            idx_le = np.insert(idx_le,0,0)
        if idx_le[-1] > idx_te[-1]:
            idx_te = np.insert(idx_te,len(idx_te),len(x)-1)
    return idx_le, idx_te
    
    
def decide_start_status(df,trip,acc_diff):
    ''' Get the reverse info based on the starting status and add manuev_init to df'''
    f = open("trip_info_beg.txt")       
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
    f = open("trip_info_end.txt")       
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
    sub = '001'
#    trip = '3133'
#    cProfile.run('read_sub(sub,trip)', 'nddatastats')
    cProfile.run('read_sub(sub)', 'nddatastats')
    p = pstats.Stats('nddatastats')
    p.sort_stats('cumulative').print_stats(10)
    
