import pandas as pd
import numpy as np

'''find the initial and end df of every trip, which speed is smaller than 15'''

sub = '058'
path = "H:\SUA_file\SUA_files\\Organized Files\\sub_"+sub+".csv"
print path

alldf = pd.read_csv(path)

d = alldf.groupby('trip', sort = False) 
df_list = [d.get_group(x) for x in d.groups]

#sort df_list based on trip number
trip_order = sorted([int(df.trip[:1]) for df in df_list])
df_list_sorted = [None]*len(trip_order)

for i in range(len(df_list)):
    for j in range(len(trip_order)):
        if int(df_list[i].trip[:1]) == trip_order[j]:
            df_list_sorted[j] = df_list[i]
            
#find maneuver information            
df_manu = pd.read_csv('N:\Raw_ND_Data\Driver_Event Query_Verififed_NADS.csv', 
                usecols=['Run ID','Type of manuever','Maneuver type end'])
manu_ID = np.array(df_manu['Run ID'])
print manu_ID
init_manu = np.array(df_manu['Type of manuever'])
end_manu = np.array(df_manu['Maneuver type end'])
            

all_df = []

for df in df_list_sorted:
    
    trip_n = int(df.trip[:1])
    print 'sub_'+str(sub), 'trip: ', trip_n
        
    speed = np.array(df.speed)
    countInit = 0
    countEnd = 0
    endIdx = 1
        
    df_cp = df.copy()      
      
    #find the init df of the trip      
    for num in speed:
        if num <15:
                countInit +=1
        else:
            break

    df_init = df_cp[:countInit]
    df_init['init?'] = '1'
    
    #add initial manuever status column
    for index in range(len(manu_ID)):
        
        manu1 = manu_ID[index]
        manu2 = 0
        
        if manu1 != manu1:
            continue
        
        #delete a,b,c in the end of tripID, like deleting a from 2715a
        if ('a'  in manu1) or ('b' in manu1) or ('c' in manu1):
            manu1 = manu_ID[index][:-1]
        
        #delete / in tripID    
        if '/' in manu1:
            manu1 = manu_ID[index][:manu_ID[index].index('/')]   
            manu2 = manu_ID[index][manu_ID[index].index('/')+1:]
            
        if (str(trip_n) == manu1) or (str(trip_n) == manu2):
            init_status = init_manu[index]
            df_init['maneuver'] = init_status
            break
            
    if 'maneuver' not in df_init.columns:
        df_init['maneuver'] = None
              
    all_df.append(df_init)
    
    #find the end df of the trip  
    while endIdx < len(speed):
        if speed[-endIdx] < 15:
            countEnd +=1
            endIdx +=1
        else:
            break
    df_end = df_cp[-countEnd:]
    df_end['init?'] = '0'
    
    #add end manuever status column
    for index in range(len(manu_ID)):
        
        manu1 = manu_ID[index]
        manu2 = 0
        
        if manu1 != manu1:
            continue
        
        #delete a,b,c in the end of tripID, like deleting a from 2715a
        if ('a' in manu1) or ('b' in manu1) or ('c' in manu1):
           # print 'manu_ID', manu_ID[index]
           #print manu_ID[index][:-1]
            manu1 = manu_ID[index][:-1]
        
        #delete / in tripID    
        if '/' in manu1:
            manu1 = manu_ID[index][:manu_ID[index].index('/')]   
            manu2 = manu_ID[index][manu_ID[index].index('/')+1:]
            
            
        if (str(trip_n) == manu1) or (str(trip_n) == manu2):
            end_status = end_manu[index]
            df_end['maneuver'] = end_status
            break
            
    if 'maneuver' not in df_end.columns:
        df_end['maneuver'] = None          
    
    
    all_df.append(df_end)

        
#combine list of frames into one dataframe
frame = pd.concat(all_df,axis=0)

#reset the order of columns
frame = frame[['subject_id','time','gpstime','latitude','longitude','heading',
                'new_heading','yaw_rate','pdop','hdop','vdop','fix_type',
                'num_sats','acc_x','acc_y','acc_z','throttle','rpm','speed',
                'Ax','trip','reverse?','manuev_init','manuev_end','init?','maneuver']]
     
#export dataframe to csv file
frame.to_csv('H:\SUA_file\SUA_files\init_end_files\sub_'+sub+'_init_end.csv', index=None)
    
    
    