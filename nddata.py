import glob
import pandas as pd
import numpy as np

PATH = r'Z:\Raw ND Data'

def read_sub(sub):
    ''' Read all the 10 hz files for one subject
    
    Arguments:
    a subject number is passed into the function as a string like: '001'
    Returns:
    The complete data frame of all a subject's runs is returned
    '''
    filenames = glob.glob(PATH + "\*" + sub + "*\*\data 10hz*.csv")
    frame = pd.DataFrame()
    dflist = []
    for filename in filenames:
        print filename
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
            continue
        if reject_file(df):
            continue   
        df = trim_file(df)
        df.run = filename[-8:-4]
        dflist.append(df)       
    frame = pd.concat(dflist,axis=0) 
    print len(frame)
    frame.to_csv('H:\SUA\SUA\Organized Files\sub_001.csv',index=None) #save all trips to one csv    
    return frame
    
def reject_file(df):
    ''' Reject a file if there is no gps movement '''
    if not(any(pd.notnull(df.gpsspeed))):
        print "gps no values"
        return True
    if max(df.gpsspeed[pd.notnull(df.gpsspeed)]) == 0:
        print "gps not moving"
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
    else:
        ismoving = df.gpsspeed > 0
        
    idx_first = np.where(ismoving)[0][0]
    idx_last = np.where(ismoving)[0][-1]
    try:
        df = df[idx_first:idx_last+1]
    except Exception:
        df = df[idx_first:idx_last] 
    if idx_last-idx_first<=600:
        df=df[0:0]  
        print "gps less than 60 sec"        
    return df   
    
if __name__ == '__main__':
    import timeit
    print(timeit.timeit("read_sub('001')", number=1, setup="from __main__ import read_sub"))
