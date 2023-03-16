import pandas as pd
import datetime
import numpy as np
from query_data import query_action_data
import time
from time import strftime, gmtime
import requests
import os
from keras.models import load_model
from sklearn.externals import joblib

pd.options.mode.chained_assignment = None
def IDFilter(df, column):
    split_df = df[column].str.split(" ", expand = True)
    MVDS = split_df.loc[split_df[2] == 'MVDS'].drop([2], axis=1)[0] + " " + split_df.loc[split_df[2] == 'MVDS'].drop([2], axis=1)[1]
    BM = split_df.loc[split_df[2] == 'BM'].drop([2], axis=1)[0] + " " + split_df.loc[split_df[2] == 'BM'].drop([2], axis=1)[1]
    df[column][list(MVDS.index)] = MVDS
    df[column][list(BM.index)] = BM
    return df

def getMajorMinor(intersection_df):
    df = []
    for key in list(['NS', 'EW']):
        if key == 'NS':
            temp = intersection_df[intersection_df['Mj_Dir'] == key]
            temp.rename(columns=dict(zip(temp.filter(regex='{}'.format(key)).columns, ['Mj_' + str(col)[0:-3] for col in temp.filter(regex='{}'.format(key)).columns])), inplace=True)
            temp.rename(columns=dict(zip(temp.filter(regex='EW').columns, ['Mi_' + str(col)[0:-3] for col in temp.filter(regex='EW').columns])), inplace=True)
            df.append(temp)
        else:
            temp = intersection_df[intersection_df['Mj_Dir'] == key]
            temp.rename(columns=dict(zip(temp.filter(regex='{}'.format(key)).columns, ['Mj_' + str(col)[0:-3] for col in temp.filter(regex='{}'.format(key)).columns])), inplace=True)
            temp.rename(columns=dict(zip(temp.filter(regex='NS').columns, ['Mi_' + str(col)[0:-3] for col in temp.filter(regex='NS').columns])), inplace=True)
            df.append(temp)
    df = pd.concat(df, ignore_index=True, sort=True)
    return df

def dotDataPreprocessing(data_dict):
    try:
        tmc_df = data_dict['TMC'][1]
        c2c_df = data_dict['c2c'][1]

        # add new column 'net_link_id'
        c2c_df['net_link_id'] = c2c_df['keySet.linkId'] + '_' + c2c_df['keySet.netId']

        # -----------------------------------  c2c group by -----------------------------------
        c2c_groupby_df = c2c_df.groupby('net_link_id').agg({'volume': 'sum', 'speed': ['mean', 'std'], 'occupancy': ['mean', 'std']})
        c2c_groupby_df.columns = ['volume', 'avg_spd', 'std_spd', 'avg_occupancy', 'std_occupancy']

        # ----------------------------------- approach group by -----------------------------------
        approach_df = tmc_df.filter(regex='(timeStamp|intersectionId|approachId|approachTMCData)')
        approach_df = approach_df.drop_duplicates(['approachId', 'timeStamp'], keep='first')

        approach_groupby_df = approach_df.groupby('approachId').agg({  'approachTMCData.totalVolume':'sum',
                                                                       'approachTMCData.greenOccupancy': ['mean', 'std'],
                                                                       'approachTMCData.redOccupancy': ['mean', 'std'],
                                                                       'approachTMCData.speed': ['mean', 'std'],
                                                                       'approachTMCData.countArrivalOnGreen': 'sum',
                                                                       'approachTMCData.countArrivalOnRed': 'sum',
                                                                       'approachTMCData.rightTurnOnRed': 'sum',
                                                                       'approachTMCData.greenTime':['sum', 'mean', 'std']
                                                                    })
        approach_groupby_df.columns = [ 'Total_volume',
                                        'avg_green_occ',
                                        'std_green_occ',
                                        'avg_red_occupancy',
                                        'std_red_occupancy',
                                        'avg_spd_ITSIQA',
                                        'std_spd_ITSIQA',
                                        'aog',
                                        'aor',
                                        'rtor',
                                        'tot_green',
                                        'avg_green',
                                        'std_green']

        # Turning Movement Count may have netative value
        approach_groupby_df[approach_groupby_df < 0] = 0
        return c2c_groupby_df, approach_groupby_df
    except BaseException:
        print("Exception occured during data pre-processing", BaseException)
        return None, None




def readBackup(obtain_time, segmentType, roadway_df, backup):
    # check the day of week is weekday or weekend
    if time.strptime(obtain_time, '%m-%d-%Y %H:%M:%S').tm_wday > 4:
        weekday_weekend = 'weekend'
    else:
        weekday_weekend = 'weekday'

    # select corresponding backup data
    backup_df = backup['backup_{}_{}'.format(segmentType, weekday_weekend)]

    # extract corresponding backup data for current minute
    minutes = time.strptime(obtain_time, '%m-%d-%Y %H:%M:%S').tm_hour*60 + time.strptime(obtain_time, '%m-%d-%Y %H:%M:%S').tm_min
    minutes_df = backup_df[backup_df['minutes'] == minutes].set_index('allId1')

    # consider five different conditions
    if segmentType == 'basic':
        roadway_df = roadway_df.sort_values('allId1').reset_index(drop=True).set_index('allId1')
        temp = roadway_df[~((roadway_df['downAvgSpd'] > 0) & (roadway_df['downVolume'] > 0) & (roadway_df['upAvgSpd'] > 0) & (roadway_df['upVolume'] > 0))]
        if len(temp) > 0:
            temp.update(minutes_df[['upVolume', 'upAvgSpd', 'upStdSpd', 'downVolume', 'downAvgSpd', 'downStdSpd']])
            roadway_df_r = pd.concat([roadway_df[(roadway_df['downAvgSpd'] > 0) & (roadway_df['downVolume'] > 0) & (roadway_df['upAvgSpd'] > 0) & (roadway_df['upVolume'] > 0)], temp]).sort_values('allId1')


    elif segmentType == 'weaving':
        roadway_df = roadway_df.sort_values('allId1').reset_index(drop=True).set_index('allId1')
        temp = roadway_df[~((roadway_df['avgSpd'] > 0) & (roadway_df['volume'] > 0) & (roadway_df['downAvgSpd'] > 0) & (roadway_df['downVolume'] > 0) & (roadway_df['upAvgSpd'] > 0) & (roadway_df['upVolume'] > 0))]
        if len(temp) > 0:
            temp.update(minutes_df[['upVolume', 'upAvgSpd', 'upStdSpd', 'downVolume', 'downAvgSpd', 'downStdSpd', 'volume','volumeVRf', 'volumeVFr']])
            roadway_df_r = pd.concat([roadway_df[(roadway_df['avgSpd'] > 0) & (roadway_df['volume'] > 0) & (roadway_df['downAvgSpd'] > 0) & (roadway_df['downVolume'] > 0) & (roadway_df['upAvgSpd'] > 0) & (roadway_df['upVolume'] > 0)], temp]).sort_values('allId1')

    elif segmentType == 'ramp':
        roadway_df = roadway_df.sort_values('allId1').reset_index(drop=True).set_index('allId1')
        temp = roadway_df[~((roadway_df['avgSpd'] > 0) & (roadway_df['stdSpd'] > 0) & (roadway_df['volume'] > 0))]
        if len(temp) > 0:
            temp.update(minutes_df[['upVolume', 'upAvgSpd', 'upStdSpd', 'downVolume', 'downAvgSpd', 'downStdSpd', 'volume', 'avgSpd', 'stdSpd']])
            roadway_df_r = pd.concat([roadway_df[(roadway_df['avgSpd'] > 0) & (roadway_df['stdSpd'] > 0) & (roadway_df['volume'] > 0)], temp]).sort_values('allId1')

    elif segmentType == 'arterial':
        roadway_df = roadway_df.sort_values('allId1').reset_index(drop=True).set_index('allId1')
        temp = roadway_df[~(roadway_df['avgSpd'] > 0)]
        if len(temp) > 0:
            temp.update(minutes_df[['avgSpd', 'stdSpd', 'upTotalVolume', 'downTotalVolume']])
            roadway_df_r = pd.concat([roadway_df[roadway_df['avgSpd'] > 0], temp]).sort_values('allId1')

    elif segmentType == 'intersection':
        roadway_df = roadway_df.sort_values('intersection').reset_index(drop=True).set_index('intersection')
        temp = roadway_df[~(roadway_df['majorAvgSpd'] > 0)]
        if len(temp) > 0:
            temp.update(minutes_df[['majorAvgSpd', 'majorStdSpd', 'minorAvgSpd', 'minorStdSpd', 'majorTotalVolume', 'minorTotalVolume']])
            roadway_df_r = pd.concat([roadway_df[roadway_df['majorAvgSpd'] > 0], temp]).sort_values('intersection')


    roadway_df_r = roadway_df_r.reset_index(drop=False)

    return roadway_df_r


def baseMapPreprocessing(base_map_df, base_map_type, obtain_time, weather_df, c2c_groupby_df, approach_groupby_df):

    # adding weather info into segment base map
    base_map_df['Weather_type'] = np.where(weather_df['weather'][0][0]['main'] == 'Rain', 1, 0)
    if 'rain' in weather_df:
        base_map_df['Precipitation'] = weather_df.filter(regex='rain').iloc[0, 0]
    else:
        base_map_df['Precipitation'] = 0
    if 'visibility' in weather_df:
        base_map_df['Visibility'] = weather_df['visibility'].iloc[0] / 1609.344
    else:
        base_map_df['Visibility'] = 10
    base_map_df['Humidity'] = weather_df['main.humidity'].iloc[0]

    # setting current timestamp and previous timestamp into segment base map
    base_map_df['curr_Timestamp'] = obtain_time
    base_map_df['prev_Timestamp'] = strftime('%m-%d-%Y %H:%M:%S', gmtime((time.mktime(time.strptime(obtain_time, '%m-%d-%Y %H:%M:%S')) - 300 - 3600 * 4)))


    # two different type of base map must join in the different way
    if base_map_type == 'intersection':
        # # remove the suffix of "MVDS" and "BM" of District 5 in base map to match c2c net ID name
        for element in ['E_app_id', 'W_app_id', 'S_app_id', 'N_app_id']:
            base_map_df = IDFilter(base_map_df, element)
        # c2c join
        base_map_df = base_map_df.join(c2c_groupby_df[['volume', 'avg_spd', 'std_spd']], on='E_app_id')
        base_map_df = base_map_df.join(c2c_groupby_df[['volume', 'avg_spd', 'std_spd']], on='W_app_id', rsuffix='_W')
        base_map_df = base_map_df.join(c2c_groupby_df[['volume', 'avg_spd', 'std_spd']], on='N_app_id', rsuffix='_N')
        base_map_df = base_map_df.join(c2c_groupby_df[['volume', 'avg_spd', 'std_spd']], on='S_app_id', rsuffix='_S')
        base_map_df = base_map_df.rename(index=str, columns={   "volume": "volume_E",
                                                                "avg_spd": "avg_spd_E",
                                                                "std_spd": "std_spd_E"})
        base_map_df['approachId_W'] = np.where(base_map_df['Intersection'].str.contains('-'), base_map_df['Intersection'].str[0:4] + base_map_df['Intersection'].str[5:9] + '-' + 'W1', base_map_df['Intersection'])
        base_map_df['approachId_S'] = np.where(base_map_df['Intersection'].str.contains('-'), base_map_df['Intersection'].str[0:4] + base_map_df['Intersection'].str[5:9] + '-' + 'S1', base_map_df['Intersection'])
        base_map_df['approachId_N'] = np.where(base_map_df['Intersection'].str.contains('-'), base_map_df['Intersection'].str[0:4] + base_map_df['Intersection'].str[5:9] + '-' + 'N1', base_map_df['Intersection'])
        base_map_df['approachId_E'] = np.where(base_map_df['Intersection'].str.contains('-'), base_map_df['Intersection'].str[0:4] + base_map_df['Intersection'].str[5:9] + '-' + 'E1', base_map_df['Intersection'])
        base_map_df = base_map_df.join(approach_groupby_df, on='approachId_E')
        base_map_df = base_map_df.join(approach_groupby_df, on='approachId_W', rsuffix='_W')
        base_map_df = base_map_df.join(approach_groupby_df, on='approachId_N', rsuffix='_N')
        base_map_df = base_map_df.join(approach_groupby_df, on='approachId_S', rsuffix='_S')
        base_map_df = base_map_df.rename(index=str, columns={"Total_volume": "Total_volume_E",
                                                             "avg_green_occ": "avg_green_occ_E",
                                                             "std_green_occ": "std_green_occ_E",
                                                             "avg_red_occupancy": "avg_red_occupancy_E",
                                                             "std_red_occupancy": "std_red_occupancy_E",
                                                             "avg_spd_ITSIQA": "avg_spd_ITSIQA_E",
                                                             "std_spd_ITSIQA": "std_spd_ITSIQA_E",
                                                             "aog": "aog_E",
                                                             "aor": "aor_E",
                                                             "rtor": "rtor_E",
                                                             "tot_green": "tot_green_E",
                                                             "avg_green": "avg_green_E",
                                                             "std_green": "std_green_E"})

    elif base_map_type == 'segment':
        # # remove the suffix of "MVDS" and "BM" of District 5 in base map to match c2c net ID name
        for element in ['ID', 'Up_segment', 'Down_segment', 'net_id']:
            base_map_df = IDFilter(base_map_df, element)
        # c2c join
        base_map_df = base_map_df.join(c2c_groupby_df, on='ID')
        base_map_df = base_map_df.join(c2c_groupby_df, on='Up_segment', rsuffix='_Up')
        base_map_df = base_map_df.join(c2c_groupby_df, on='Down_segment', rsuffix='_Down')
        base_map_df = base_map_df.join(c2c_groupby_df['volume'], on='Weav_on', rsuffix='_V_rf')
        base_map_df = base_map_df.join(c2c_groupby_df['volume'], on='Weav_off', rsuffix='_V_fr')

        # approach join
        base_map_df['Up_approach'] = np.where(base_map_df['Up_signal'].str.contains('-'), base_map_df['Up_signal'].str[0:4] + base_map_df['Up_signal'].str[5:9] + '-' + base_map_df['direction'].str[0:1] + '1', 0)
        base_map_df['Down_approach'] = np.where(base_map_df['Down_signal'].str.contains('-'), base_map_df['Down_signal'].str[0:4] + base_map_df['Down_signal'].str[5:9] + '-' + base_map_df['direction'].str[0:1] + '1', 0)
        base_map_df = base_map_df.join(approach_groupby_df, on='Up_approach')
        base_map_df = base_map_df.join(approach_groupby_df, on='Down_approach', rsuffix='_Down')
        base_map_df = base_map_df.rename(index=str, columns={  "Total_volume": "Total_volume_Up",
                                                               "avg_green_occ": "avg_green_occ_Up",
                                                               "std_green_occ": "std_green_occ_Up",
                                                               "avg_red_occupancy": "avg_red_occupancy_Up",
                                                               "std_red_occupancy": "std_red_occupancy_Up",
                                                               "avg_spd_ITSIQA": "avg_spd_ITSIQA_Up",
                                                               "std_spd_ITSIQA": "std_spd_ITSIQA_Up",
                                                               "aog": "aog_Up",
                                                               "aor": "aor_Up",
                                                               "rtor": "rtor_Up",
                                                               "tot_green": "tot_green_Up",
                                                               "avg_green": "avg_green_Up",
                                                               "std_green": "std_green_Up"})
    else:
        return 'base map type error.'

    return base_map_df


def readBackupFile():
    backup = {}
    backup['backup_arterial_weekday'] = pd.read_csv('backup/backup_arterial_weekday.csv', index_col = 0)
    # backup['backup_arterial_weekday']['allId1'] = backup['backup_arterial_weekday']['allId1'].astype(str)

    backup['backup_arterial_weekend'] = pd.read_csv('backup/backup_arterial_weekend.csv', index_col = 0)
    # backup['backup_arterial_weekend']['allId1'] = backup['backup_arterial_weekend']['allId1'].astype(str)

    backup['backup_basic_weekday'] = pd.read_csv('backup/backup_basic_weekday.csv', index_col = 0)
    # backup['backup_basic_weekday']['allId1'] = backup['backup_basic_weekday']['allId1'].astype(str)

    backup['backup_basic_weekend'] = pd.read_csv('backup/backup_basic_weekend.csv', index_col = 0)
    # backup['backup_basic_weekend']['allId1'] = backup['backup_basic_weekend']['allId1'].astype(str)

    backup['backup_intersection_weekday'] = pd.read_csv('backup/backup_intersection_weekday.csv', index_col = 0)
    backup['backup_intersection_weekday']['allId1'] = backup['backup_intersection_weekday']['allId1'].astype(str)

    backup['backup_intersection_weekend'] = pd.read_csv('backup/backup_intersection_weekend.csv', index_col = 0)
    backup['backup_intersection_weekend']['allId1'] = backup['backup_intersection_weekend']['allId1'].astype(str)

    backup['backup_ramp_weekday'] = pd.read_csv('backup/backup_ramp_weekday.csv', index_col = 0)
    # backup['backup_ramp_weekday']['allId1'] = backup['backup_ramp_weekday']['allId1'].astype(str)

    backup['backup_ramp_weekend'] = pd.read_csv('backup/backup_ramp_weekend.csv', index_col = 0)
    # backup['backup_ramp_weekend']['allId1'] = backup['backup_ramp_weekend']['allId1'].astype(str)

    backup['backup_weaving_weekday'] = pd.read_csv('backup/backup_weaving_weekday.csv', index_col = 0)
    # backup['backup_weaving_weekday']['allId1'] = backup['backup_weaving_weekday']['allId1'].astype(str)

    backup['backup_weaving_weekend'] = pd.read_csv('backup/backup_weaving_weekend.csv', index_col = 0)
    # backup['backup_weaving_weekend']['allId1'] = backup['backup_weaving_weekend']['allId1'].astype(str)

    return backup

def freewayBasicSegment(freeway_basic_segment_df, obtain_time, backup):
    # rename variables
    freeway_basic_segment_df = freeway_basic_segment_df.rename(index=str, columns={   'ID':'linkNetId',
                                                                                      'lane_count':'laneCount',
                                                                                      'spd_limit':'spdLimit',
                                                                                      'net_id':'netId',
                                                                                      'Seg_Type':'segType',
                                                                                      'Up_segment':'upSegment',
                                                                                      'Down_segment':'downSegment',
                                                                                      'Sub_ID':'subId',
                                                                                      'Roadway':'roadway',
                                                                                      'All_ID1':'allId1',
                                                                                      'End':'end',
                                                                                      'Weather_type':'weatherType',
                                                                                      'Precipitation':'precipitation',
                                                                                      'Visibility':'visibility',
                                                                                      'Humidity':'humidity',
                                                                                      'curr_Timestamp':'currTimestamp',
                                                                                      'prev_Timestamp':'prevTimestamp',
                                                                                      'avg_spd':'avgSpd',
                                                                                      'std_spd':'stdSpd',
                                                                                      'avg_occupancy':'avgOccupancy',
                                                                                      'std_occupancy':'stdOccupancy',
                                                                                      'volume_Up':'upVolume',
                                                                                      'avg_spd_Up':'upAvgSpd',
                                                                                      'std_spd_Up':'upStdSpd',
                                                                                      'avg_occupancy_Up':'upAvgOccupancy',
                                                                                      'std_occupancy_Up':'upStdOccupancy',
                                                                                      'volume_Down':'downVolume',
                                                                                      'avg_spd_Down':'downAvgSpd',
                                                                                      'std_spd_Down':'downStdSpd',
                                                                                      'avg_occupancy_Down':'downAvgOccupancy',
                                                                                      'std_occupancy_Down':'downStdOccupancy'})

    # integrate data with corresponding backup data
    freeway_basic_segment_df = readBackup(obtain_time, 'basic', freeway_basic_segment_df, backup)

    # congest index
    freeway_basic_segment_df['downCongestIndex'] = np.where(freeway_basic_segment_df['downAvgSpd'] > 0, ((freeway_basic_segment_df['spdLimit'] - freeway_basic_segment_df['downAvgSpd']) / freeway_basic_segment_df['spdLimit']).clip(lower=0.0), 0)
    freeway_basic_segment_df['upCongestIndex'] = np.where(freeway_basic_segment_df['upAvgSpd'] > 0, ((freeway_basic_segment_df['spdLimit'] - freeway_basic_segment_df['upAvgSpd']) / freeway_basic_segment_df['spdLimit']).clip(lower=0.0), 0)
    freeway_basic_segment_df['congestIndex'] = np.where(freeway_basic_segment_df['avgSpd'] > 0, ((freeway_basic_segment_df['spdLimit'] - freeway_basic_segment_df['avgSpd']) / freeway_basic_segment_df['spdLimit']).clip(lower=0.0), 0)
    # freeway_basic_segment_df['Up_congest_index'] = ((freeway_basic_segment_df['spd_limit'] - freeway_basic_segment_df['avg_spd_Up']) / freeway_basic_segment_df['spd_limit']).clip(lower=0.0)
    # freeway_basic_segment_df['Congest_index'] = ((freeway_basic_segment_df['spd_limit'] - freeway_basic_segment_df['avg_spd']) / freeway_basic_segment_df['spd_limit']).clip(lower=0.0)

    # calculate crash risk
    var = np.exp(   (-1.505) +
                    0.382 * np.log(freeway_basic_segment_df['upVolume']) +
                    (-0.042) * freeway_basic_segment_df['upAvgSpd'] +
                    6.809 * freeway_basic_segment_df['downCongestIndex'])
    risk = var / (1 + var)
    freeway_basic_segment_df['crashRisk'] = ((1/4000)*risk*100)/(1 - risk + (1/4000)*risk)
    freeway_basic_segment_df['crashRisk'] = freeway_basic_segment_df['crashRisk'].round(5)
    # calculate severe crash risk
    var_ = np.exp(  (-6.06335) +
                    0.039313 * freeway_basic_segment_df['downAvgSpd'])
    freeway_basic_segment_df['severeCrashRisk'] = (var_/ (1 + var_)) * freeway_basic_segment_df['crashRisk']

    freeway_basic_segment_df=freeway_basic_segment_df.fillna(0)
    return freeway_basic_segment_df.drop(columns=['FID','start_node','end_node','Rmp_Type','Checked','P','Up_signal','Down_signal','All_ID','Up_consist','Down_consist','check','Weav_LC','N_WL','Weav_on','Weav_off','volume_V_rf','volume_V_fr','Up_approach','Down_approach','Total_volume_Up','avg_green_occ_Up','std_green_occ_Up','avg_red_occupancy_Up','std_red_occupancy_Up','avg_spd_ITSIQA_Up','std_spd_ITSIQA_Up','aog_Up','aor_Up','rtor_Up','tot_green_Up','avg_green_Up','std_green_Up','Total_volume_Down','avg_green_occ_Down','std_green_occ_Down','avg_red_occupancy_Down','std_red_occupancy_Down','avg_spd_ITSIQA_Down','std_spd_ITSIQA_Down','aog_Down','aor_Down','rtor_Down','tot_green_Down','avg_green_Down','std_green_Down'])


def freewayWeavingSegment(freeway_weaving_segment_df, obtain_time, backup):
    # rename variables
    freeway_weaving_segment_df = freeway_weaving_segment_df.rename(index=str, columns={ 'ID':'linkNetId',
                                                                                        'lane_count':'laneCount',
                                                                                        'spd_limit':'spdLimit',
                                                                                        'net_id':'netId',
                                                                                        'Seg_Type':'segType',
                                                                                        'Up_segment':'upSegment',
                                                                                        'Down_segment':'downSegment',
                                                                                        'Sub_ID':'subId',
                                                                                        'Roadway':'roadway',
                                                                                        'All_ID1':'allId1',
                                                                                        'Weav_LC':'weaveLC',
                                                                                        'N_WL':'nwl',
                                                                                        'Weav_on':'weaveOn',
                                                                                        'Weav_off':'weaveOff',
                                                                                        'Weather_type':'weatherType',
                                                                                        'Precipitation':'precipitation',
                                                                                        'Visibility':'visibility',
                                                                                        'Humidity':'humidity',
                                                                                        'curr_Timestamp':'currTimestamp',
                                                                                        'prev_Timestamp':'prevTimestamp',
                                                                                        'avg_spd':'avgSpd',
                                                                                        'std_spd':'stdSpd',
                                                                                        'avg_occupancy':'avgOccupancy',
                                                                                        'std_occupancy':'stdOccupancy',
                                                                                        'volume_Up':'upVolume',
                                                                                        'avg_spd_Up':'upAvgSpd',
                                                                                        'std_spd_Up':'upStdSpd',
                                                                                        'avg_occupancy_Up':'upAvgOccupancy',
                                                                                        'std_occupancy_Up':'upStdOccupancy',
                                                                                        'volume_Down':'downVolume',
                                                                                        'avg_spd_Down':'downAvgSpd',
                                                                                        'std_spd_Down':'downStdSpd',
                                                                                        'avg_occupancy_Down':'downAvgOccupancy',
                                                                                        'std_occupancy_Down':'downStdOccupancy',
                                                                                        'volume_V_rf':'volumeVRf',
                                                                                        'volume_V_fr':'volumeVFr'})

    # integrate data with corresponding backup data
    freeway_weaving_segment_df = readBackup(obtain_time, 'weaving', freeway_weaving_segment_df, backup)

    freeway_weaving_segment_df['spdDiff'] =  freeway_weaving_segment_df['upAvgSpd'] - freeway_weaving_segment_df['downAvgSpd']
    freeway_weaving_segment_df['volumeVRf'] = freeway_weaving_segment_df['volumeVRf'].fillna(0)
    freeway_weaving_segment_df['volumeVFr'] = freeway_weaving_segment_df['volumeVFr'].fillna(0)
    freeway_weaving_segment_df['volume'] = freeway_weaving_segment_df['volume'].fillna(0)

    var = freeway_weaving_segment_df['volumeVRf'] + freeway_weaving_segment_df['volumeVFr']
    freeway_weaving_segment_df['vr'] = var / (freeway_weaving_segment_df['volume'] + var)
    freeway_weaving_segment_df['maxWeavingLen'] = (5728 * (1 + freeway_weaving_segment_df['vr']) ** 1.6 - 1566 * freeway_weaving_segment_df['nwl']) / 1000


    var = np.exp(   (-0.111) * freeway_weaving_segment_df['upAvgSpd'] +
                     0.064 * freeway_weaving_segment_df['spdDiff'] +
                     0.536 * np.log(freeway_weaving_segment_df['volumeVRf'] + freeway_weaving_segment_df['volumeVFr'] + freeway_weaving_segment_df['volume']) +
                     0.571 * freeway_weaving_segment_df['weatherType'] +
                     0.3 * freeway_weaving_segment_df['maxWeavingLen'] +
                     0.702 * freeway_weaving_segment_df['weaveLC'])
    risk = var / (1 + var)
    freeway_weaving_segment_df['crashRisk'] = ((1/4000)*risk*100)/(1 - risk + (1/4000)*risk)
    freeway_weaving_segment_df['crashRisk'] = freeway_weaving_segment_df['crashRisk'].round(5)

    # weav severe crash risk
    var_ = np.exp( (-6.06335) +
                   0.039313 * freeway_weaving_segment_df['downAvgSpd'])
    freeway_weaving_segment_df['severeCrashRisk'] = (var_/ (1 + var_)) * freeway_weaving_segment_df['crashRisk']

    freeway_weaving_segment_df=freeway_weaving_segment_df.fillna(0)
    return freeway_weaving_segment_df.drop(columns=['FID','start_node','end_node','Rmp_Type','Checked','P','Up_signal','Down_signal','All_ID','Up_consist','Down_consist','check','End','Up_approach','Down_approach','Total_volume_Up','avg_green_occ_Up','std_green_occ_Up','avg_red_occupancy_Up','std_red_occupancy_Up','avg_spd_ITSIQA_Up','std_spd_ITSIQA_Up','aog_Up','aor_Up','rtor_Up','tot_green_Up','avg_green_Up','std_green_Up','Total_volume_Down','avg_green_occ_Down','std_green_occ_Down','avg_red_occupancy_Down','std_red_occupancy_Down','avg_spd_ITSIQA_Down','std_spd_ITSIQA_Down','aog_Down','aor_Down','rtor_Down','tot_green_Down','avg_green_Down','std_green_Down'])

def freewayRampSegment(freeway_ramp_segment_df, obtain_time, backup):
    # rename variables
    freeway_ramp_segment_df = freeway_ramp_segment_df.rename(index=str, columns={   'ID':'linkNetId',
                                                                                    'lane_count':'laneCount',
                                                                                    'spd_limit':'spdLimit',
                                                                                    'net_id':'netId',
                                                                                    'Seg_Type':'segType',
                                                                                    'Up_segment':'upSegment',
                                                                                    'Down_segment':'downSegment',
                                                                                    'Sub_ID':'subId',
                                                                                    'Roadway':'roadway',
                                                                                    'All_ID1':'allId1',
                                                                                    'Weather_type':'weatherType',
                                                                                    'Precipitation':'precipitation',
                                                                                    'Visibility':'visibility',
                                                                                    'Humidity':'humidity',
                                                                                    'curr_Timestamp':'currTimestamp',
                                                                                    'prev_Timestamp':'prevTimestamp',
                                                                                    'volume':'volume',
                                                                                    'avg_spd':'avgSpd',
                                                                                    'std_spd':'stdSpd',
                                                                                    'avg_occupancy':'avgOccupancy',
                                                                                    'std_occupancy':'stdOccupancy',
                                                                                    'volume_Up':'upVolume',
                                                                                    'avg_spd_Up':'upAvgSpd',
                                                                                    'std_spd_Up':'upStdSpd',
                                                                                    'avg_occupancy_Up':'upAvgOccupancy',
                                                                                    'std_occupancy_Up':'upStdOccupancy',
                                                                                    'volume_Down':'downVolume',
                                                                                    'avg_spd_Down':'downAvgSpd',
                                                                                    'std_spd_Down':'downStdSpd',
                                                                                    'avg_occupancy_Down':'downAvgOccupancy',
                                                                                    'std_occupancy_Down':'downStdOccupancy'})

    # integrate data with corresponding backup data
    freeway_ramp_segment_df = readBackup(obtain_time, 'ramp', freeway_ramp_segment_df, backup)

    freeway_ramp_segment_df['rmpTypeCode'] = np.where(freeway_ramp_segment_df['Rmp_Type'] == "OFF", 1, 0)
    # crash risk for ramp
    var = np.exp(   (-8.959) +
                    1.157 * np.log( freeway_ramp_segment_df['volume'] ) +
                    0.048 * freeway_ramp_segment_df['avgSpd'] +
                    0.065 * freeway_ramp_segment_df['stdSpd'] +
                    0.845 * freeway_ramp_segment_df['rmpTypeCode'] +
                    (-0.147) * freeway_ramp_segment_df['visibility'])

    risk = var / (1 + var)
    freeway_ramp_segment_df['crashRisk'] = ((1/4000)*risk*100)/(1 - risk + (1/4000)*risk)
    freeway_ramp_segment_df['crashRisk'] = freeway_ramp_segment_df['crashRisk'].round(5)
    # ramp severe crash risk
    var_ = np.exp( (-6.06335) +
                   0.039313 * freeway_ramp_segment_df['downAvgSpd'])
    freeway_ramp_segment_df['severeCrashRisk'] = (var_/ (1 + var_)) * freeway_ramp_segment_df['crashRisk']

    freeway_ramp_segment_df=freeway_ramp_segment_df.fillna(0)
    return freeway_ramp_segment_df.drop(columns=['FID','start_node','end_node','Rmp_Type','Checked','P','Up_signal','Down_signal','All_ID','Up_consist','Down_consist','check','End','Weav_LC','N_WL','Weav_on','Weav_off','volume_V_rf','volume_V_fr','Up_approach','Down_approach','Total_volume_Up','avg_green_occ_Up','std_green_occ_Up','avg_red_occupancy_Up','std_red_occupancy_Up','avg_spd_ITSIQA_Up','std_spd_ITSIQA_Up','aog_Up','aor_Up','rtor_Up','tot_green_Up','avg_green_Up','std_green_Up','Total_volume_Down','avg_green_occ_Down','std_green_occ_Down','avg_red_occupancy_Down','std_red_occupancy_Down','avg_spd_ITSIQA_Down','std_spd_ITSIQA_Down','aog_Down','aor_Down','rtor_Down','tot_green_Down','avg_green_Down','std_green_Down'])

def arterialSegment(arterial_df, obtain_time, backup):
    # rename variables
    arterial_df = arterial_df.rename(index=str, columns={   'ID':'linkNetId',
                                                            'lane_count':'laneCount',
                                                            'spd_limit':'spdLimit',
                                                            'net_id':'netId',
                                                            'Seg_Type':'segType',
                                                            'Up_signal':'upSignal',
                                                            'Down_signal':'downSignal',
                                                            'Sub_ID':'subId',
                                                            'All_ID1':'allId1',
                                                            'Weather_type':'weatherType',
                                                            'Precipitation':'precipitation',
                                                            'Visibility':'visibility',
                                                            'Humidity':'humidity',
                                                            'curr_Timestamp':'currTimestamp',
                                                            'prev_Timestamp':'prevTimestamp',
                                                            'avg_spd':'avgSpd',
                                                            'std_spd':'stdSpd',
                                                            'avg_occupancy':'avgOccupancy',
                                                            'std_occupancy':'stdOccupancy',
                                                            'Up_approach':'upApproach',
                                                            'Down_approach':'downApproach',
                                                            'Total_volume_Up':'upTotalVolume',
                                                            'avg_green_occ_Up':'upAvgGreenOccupancy',
                                                            'std_green_occ_Up':'upStdGreenwOccupancy',
                                                            'avg_red_occupancy_Up':'upAvgRedOccupancy',
                                                            'std_red_occupancy_Up':'upStdRedOccupancy',
                                                            'avg_spd_ITSIQA_Up':'upAvgSpdITSIQA',
                                                            'std_spd_ITSIQA_Up':'upStdSpdITSIQA',
                                                            'aog_Up':'upAog',
                                                            'aor_Up':'upAor',
                                                            'rtor_Up':'upRtor',
                                                            'tot_green_Up':'upTotalGreen',
                                                            'avg_green_Up':'upAvgGreen',
                                                            'std_green_Up':'upStdGreen',
                                                            'Total_volume_Down':'downTotalVolume',
                                                            'avg_green_occ_Down':'downAvgGreenOccupancy',
                                                            'std_green_occ_Down':'downStdGreenOccupancy',
                                                            'avg_red_occupancy_Down':'downAvgRedOccupancy',
                                                            'std_red_occupancy_Down':'downStdRedOccupancy',
                                                            'avg_spd_ITSIQA_Down':'downAvgSpdITSIQA',
                                                            'std_spd_ITSIQA_Down':'downStdSpdITSIQA',
                                                            'aog_Down':'downAog',
                                                            'aor_Down':'downAor',
                                                            'rtor_Down':'downRtor',
                                                            'tot_green_Down':'downTotalGreen',
                                                            'avg_green_Down':'downAvgGreen',
                                                            'std_green_Down':'downStdGreen'})

    # integrate data with corresponding backup data
    arterial_df = readBackup(obtain_time, 'arterial', arterial_df, backup)

    arterial_df['upGreenRatio'] = arterial_df['upTotalGreen'] / 300
    arterial_df['downGreenRatio'] = arterial_df['downTotalGreen'] / 300
    arterial_df['upAoy'] = arterial_df['upTotalVolume'] - arterial_df['upAog'] - arterial_df['upAor']
    arterial_df['upAoy'] = np.where(arterial_df['upSignal'].str.contains('SEM'), 0, arterial_df['upAoy'])
    arterial_df['downAoy'] = arterial_df['downTotalVolume'] - arterial_df['downAog'] - arterial_df['downAor']
    arterial_df['downAoy'] = np.where(arterial_df['downSignal'].str.contains('SEM'), 0, arterial_df['downAoy'])

    arterial_df['downGreenRatio']=arterial_df['downGreenRatio'].fillna(1)
    arterial_df['upGreenRatio']=arterial_df['upGreenRatio'].fillna(1)
    arterial_df=arterial_df.fillna(0)
    var = np.exp(    (-0.8381) +
                 (-0.0222) * arterial_df['avgSpd'] +
                 0.034 * arterial_df['downAoy'] +
                 (-0.06893) * arterial_df['downGreenRatio'] +
                 0.0038 * arterial_df['upTotalVolume'] +
                 (-0.5045) * arterial_df['upGreenRatio'] +
                 (-0.0029) * arterial_df['humidity'] +
                 (-0.0212) * arterial_df['stdSpd'] +
                 (-0.3549) * arterial_df['weatherType'] +
                 (-0.0405) * arterial_df['visibility'])

    risk = var / (1 + var)
    arterial_df['crashRisk'] =((1/4000)*risk*100)/(1 - risk + (1/4000)*risk)
    arterial_df['crashRisk'] = arterial_df['crashRisk'].round(5)
    var_ = np.exp( (-5.8556) +
                0.0512 * arterial_df['avgSpd'])
    arterial_df['severeCrashRisk'] = (var_/ (1 + var_)) * arterial_df['crashRisk']

    arterial_df=arterial_df.fillna(0)
    return arterial_df.drop(columns=['FID', 'start_node', 'end_node','Up_segment','Down_segment','Rmp_Type','Checked','P','All_ID','Roadway','Up_consist','Down_consist','check','End','Weav_LC','N_WL','Weav_on','Weav_off','volume_V_rf','volume_V_fr','volume_Up','avg_spd_Up','std_spd_Up','avg_occupancy_Up','std_occupancy_Up','volume_Down','avg_spd_Down','std_spd_Down','avg_occupancy_Down','std_occupancy_Down'])

def intersection(intersection_df, obtain_time, backup):
    intersection_df['volume_EW'] = intersection_df['volume_E'] + intersection_df['volume_W']
    intersection_df['avg_spd_EW'] = (intersection_df['avg_spd_E'] + intersection_df['avg_spd_W']) / 2
    intersection_df['std_spd_EW'] = (intersection_df['std_spd_E'] + intersection_df['std_spd_W']) / 2

    intersection_df['volume_NS'] = intersection_df['volume_N'] + intersection_df['volume_S']
    intersection_df['avg_spd_NS'] = (intersection_df['avg_spd_N'] + intersection_df['avg_spd_S']) / 2
    intersection_df['std_spd_NS'] = (intersection_df['std_spd_N'] + intersection_df['std_spd_S']) / 2


    intersection_df['Total_volume_NS'] = intersection_df['Total_volume_N'] + intersection_df['Total_volume_S']
    intersection_df['avg_green_occ_NS'] = (intersection_df['avg_green_occ_N'] + intersection_df['avg_green_occ_S']) / 2
    intersection_df['std_green_occ_NS'] = (intersection_df['std_green_occ_N'] + intersection_df['std_green_occ_S']) / 2
    intersection_df['avg_red_occupancy_NS'] = (intersection_df['avg_red_occupancy_N'] + intersection_df['avg_red_occupancy_S']) / 2
    intersection_df['std_red_occupancy_NS'] = (intersection_df['std_red_occupancy_N'] + intersection_df['std_red_occupancy_S']) / 2
    intersection_df['avg_spd_ITSIQA_NS'] = (intersection_df['avg_spd_ITSIQA_N'] + intersection_df['avg_spd_ITSIQA_S']) / 2
    intersection_df['std_spd_ITSIQA_NS'] = (intersection_df['std_spd_ITSIQA_N'] + intersection_df['std_spd_ITSIQA_S']) / 2
    intersection_df['aog_NS'] = intersection_df['aog_N'] + intersection_df['aog_S']
    intersection_df['aor_NS'] = intersection_df['aor_N'] + intersection_df['aor_S']
    intersection_df['rtor_NS'] = intersection_df['rtor_N'] + intersection_df['rtor_S']
    intersection_df['tot_green_NS'] = (intersection_df['tot_green_N'] + intersection_df['tot_green_S']) / 2
    intersection_df['avg_green_NS'] = (intersection_df['avg_green_N'] + intersection_df['avg_green_S']) / 2
    intersection_df['std_green_NS'] = (intersection_df['std_green_N'] + intersection_df['std_green_S']) / 2

    intersection_df['Total_volume_EW'] = intersection_df['Total_volume_E'] + intersection_df['Total_volume_W']
    intersection_df['avg_green_occ_EW'] = (intersection_df['avg_green_occ_E'] + intersection_df['avg_green_occ_W']) / 2
    intersection_df['std_green_occ_EW'] = (intersection_df['std_green_occ_E'] + intersection_df['std_green_occ_W']) / 2
    intersection_df['avg_red_occupancy_EW'] = (intersection_df['avg_red_occupancy_E'] + intersection_df['avg_red_occupancy_W']) / 2
    intersection_df['std_red_occupancy_EW'] = (intersection_df['std_red_occupancy_E'] + intersection_df['std_red_occupancy_W']) / 2
    intersection_df['avg_spd_ITSIQA_EW'] = (intersection_df['avg_spd_ITSIQA_E'] + intersection_df['avg_spd_ITSIQA_W']) / 2
    intersection_df['std_spd_ITSIQA_EW'] = (intersection_df['std_spd_ITSIQA_E'] + intersection_df['std_spd_ITSIQA_W']) / 2
    intersection_df['aog_EW'] = intersection_df['aog_E'] + intersection_df['aog_W']
    intersection_df['aor_EW'] = intersection_df['aor_E'] + intersection_df['aor_W']
    intersection_df['rtor_EW'] = intersection_df['rtor_E'] + intersection_df['rtor_W']
    intersection_df['tot_green_EW'] = (intersection_df['tot_green_E'] + intersection_df['tot_green_W']) / 2
    intersection_df['avg_green_EW'] = (intersection_df['avg_green_E'] + intersection_df['avg_green_W']) / 2
    intersection_df['std_green_EW'] = (intersection_df['std_green_E'] + intersection_df['std_green_W']) / 2

    drop_list = []
    for key in intersection_df:
        if "approachId" in key:
            continue
        if key[-2:] == "_S" or key[-2:] == "_N" or key[-2:] == "_E" or key[-2:] == "_W":
            drop_list.append(key)

    intersection_df = intersection_df.drop(columns=drop_list)
    intersection_df = getMajorMinor(intersection_df)

    # rename variables
    intersection_df = intersection_df.rename(index=str, columns={   'County':'county',
                                                                    'E_app_id':'eAppId',
                                                                    'Humidity':'humidity',
                                                                    'Intersection':'intersection',
                                                                    'Latitude':'latitude',
                                                                    'Longitude':'longitude',
                                                                    'Mi_Total_volume':'minorTotalVolume',
                                                                    'Mi_aog':'minorAog',
                                                                    'Mi_aor':'minorAor',
                                                                    'Mi_avg_green':'minorAvgGreen',
                                                                    'Mi_avg_green_occ':'minorAvgGreenOccpancy',
                                                                    'Mi_avg_red_occupancy':'minorAvgRedOccupancy',
                                                                    'Mi_avg_spd':'minorAvgSpd',
                                                                    'Mi_avg_spd_ITSIQA':'minorAvgSpdITSIQA',
                                                                    'Mi_rtor':'minorRtor',
                                                                    'Mi_std_green':'minorStdGreen',
                                                                    'Mi_std_green_occ':'minorStdGreenOccupancy',
                                                                    'Mi_std_red_occupancy':'minorStdRedOccupancy',
                                                                    'Mi_std_spd':'minorStdSpd',
                                                                    'Mi_std_spd_ITSIQA':'minorStdSpdITSIQA',
                                                                    'Mi_tot_green':'minorTotalGreen',
                                                                    'Mi_volume':'minorVolume',
                                                                    'Mj_Dir':'majorDir',
                                                                    'Mj_Total_volume':'majorTotalVolume',
                                                                    'Mj_aog':'majorAog',
                                                                    'Mj_aor':'majorAor',
                                                                    'Mj_avg_green':'majorAvgGreen',
                                                                    'Mj_avg_green_occ':'majorAvgGreenOccupancy',
                                                                    'Mj_avg_red_occupancy':'majorAvgRedOccupancy',
                                                                    'Mj_avg_spd':'majorAvgSpd',
                                                                    'Mj_avg_spd_ITSIQA':'majorAvgSpdITSIQA',
                                                                    'Mj_rtor':'majorRtor',
                                                                    'Mj_std_green':'majorStdGreen',
                                                                    'Mj_std_green_occ':'majorStdGreenOccupancy',
                                                                    'Mj_std_red_occupancy':'majorStdRedOccupancy',
                                                                    'Mj_std_spd':'majorStdSpd',
                                                                    'Mj_std_spd_ITSIQA':'majorStdSpdITSIQA',
                                                                    'Mj_tot_green':'majorTotalGreen',
                                                                    'Mj_volume':'majorVolume',
                                                                    'N_app_id':'nAppId',
                                                                    'Name':'name',
                                                                    'Precipitation':'precipitation',
                                                                    'Road':'road',
                                                                    'S_app_id':'sAppId',
                                                                    'TimeStamp':'timeStamp',
                                                                    'Visibility':'visibility',
                                                                    'W_app_id':'wAppId',
                                                                    'Weather_type':'weatherType',
                                                                    'approachId_E':'eApproachId',
                                                                    'approachId_N':'nApproachId',
                                                                    'approachId_S':'sApproachId',
                                                                    'approachId_W':'wApproachId',
                                                                    'curr_Timestamp':'currTimestamp',
                                                                    'prev_Timestamp':'prevTimestamp'})

    # integrate data with corresponding backup data
    intersection_df = readBackup(obtain_time, 'intersection', intersection_df, backup)


    intersection_df['majorTotalVolume'] = np.where(intersection_df['intersection'].str.contains('-'), intersection_df['majorTotalVolume'], intersection_df['majorVolume'] )
    intersection_df['minorTotalVolume'] = np.where(intersection_df['intersection'].str.contains('-'), intersection_df['minorTotalVolume'], intersection_df['minorVolume'] )

    intersection_df['majorAvgSpd'] = np.where((intersection_df['intersection'].str.contains('ORL|ORA')) & (intersection_df['majorAvgSpdITSIQA'] > 0), intersection_df['majorAvgSpdITSIQA'], intersection_df['majorAvgSpd'] )
    intersection_df['minorAvgSpd'] = np.where((intersection_df['intersection'].str.contains('ORL|ORA')) & (intersection_df['minorAvgSpdITSIQA'] > 0), intersection_df['minorAvgSpdITSIQA'], intersection_df['minorAvgSpd'] )

    intersection_df['majorStdSpd'] = np.where((intersection_df['intersection'].str.contains('ORL|ORA')) & (intersection_df['majorStdSpdITSIQA'] > 0), intersection_df['majorStdSpdITSIQA'], intersection_df['majorStdSpd'] )
    intersection_df['minorStdSpd'] = np.where((intersection_df['intersection'].str.contains('ORL|ORA')) & (intersection_df['minorStdSpdITSIQA'] > 0), intersection_df['minorStdSpdITSIQA'], intersection_df['minorStdSpd'] )

    intersection_df['majorGreenRatio'] = intersection_df['majorTotalGreen'] / 300
    intersection_df['minorGreenRatio'] = intersection_df['minorTotalGreen'] / 300

    intersection_df['majorAoy'] = intersection_df['majorTotalVolume'] - intersection_df['majorAog'] - intersection_df['majorAor']
    intersection_df['majorAoy'] = np.where(intersection_df['intersection'].str.contains('SEM'), 0, intersection_df['majorAoy'])
    intersection_df['minorAoy'] = intersection_df['minorTotalVolume'] - intersection_df['minorAog'] - intersection_df['minorAor']
    intersection_df['minorAoy'] = np.where(intersection_df['intersection'].str.contains('SEM'), 0, intersection_df['minorAoy'])


    intersection_df['majorGreenRatio']=intersection_df['majorGreenRatio'].fillna(1)
    intersection_df['minorGreenRatio']=intersection_df['minorGreenRatio'].fillna(1)
    intersection_df=intersection_df.fillna(0)
    # crash risk
    var = np.exp( 0.2301 +
                 (-0.0303) * intersection_df['majorAvgSpd'] +
                 0.0294 * intersection_df['majorStdSpd'] +
                 (-0.0542)* intersection_df['visibility'] +
                 (-0.004) * intersection_df['humidity'] +
                 1.1842 * intersection_df['precipitation'] +
                 0.0016 * intersection_df['majorTotalVolume'] +
                 0.0218 * intersection_df['majorAoy'] +
                 (-0.6491) * intersection_df['majorGreenRatio'] +
                 0.0014 * intersection_df['minorTotalVolume'] +
                 (-0.4920) * intersection_df['minorGreenRatio'])

    risk = var / (1 + var)
    intersection_df['crashRisk'] = ((1/4000)*risk*100)/(1 - risk + (1/4000)*risk)
    intersection_df['crashRisk'] = intersection_df['crashRisk'].round(5)
    var_ = np.exp( (-2.3455) +
                   (-1.1262) * intersection_df['minorGreenRatio'])
    intersection_df['severeCrashRisk'] = (var_/ (1 + var_)) * intersection_df['crashRisk']


    return intersection_df.drop(columns=['DataOutput','FID','FID_','Field1','P'])


def upload2DB(df, Ip, numberOfSubmit):
    data_list = []
    for i in range(len(df)):
        data_dict = {}
        for column in df:
            if df[column].dtype == 'float32' or df[column].dtype == 'float64':
                data_dict[column] = float(df[column][i])
            elif df[column].dtype == 'int32' or df[column].dtype == 'int64':
                data_dict[column] = int(df[column][i])
            else:
                data_dict[column] = df[column][i]
        data_list.append(data_dict)
        if len(data_list) >= numberOfSubmit:
            r = requests.post(Ip, json=data_list)
            if  r.status_code != 200:
                print('Roadway Filter:', time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime(time.time())) ,' Upload file to', Ip, 'error, error code number:', r.status_code)
            # print('Upload', numberOfSubmit, 'files to', Ip, 'at', datetime.datetime.now(), r.status_code)
            data_list.clear()
    if len(data_list) != 0:
        r = requests.post(Ip, json=data_list)
        if  r.status_code != 200:
            print('Roadway Filter:', time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime(time.time())) ,' Upload file to', Ip, 'error, error code number:', r.status_code)
        # print('Upload', len(data_list), 'files to', Ip, 'at', datetime.datetime.now(), r.status_code)
        data_list.clear()

def lstmPredicting(datasetDict, segType, model):
    for i in range(len(datasetDict[segType])):
        if len(datasetDict[segType][i]) == 4:
            if segType == 'arterial':
                dataset = np.asarray([ datasetDict[segType][i][0], datasetDict[segType][i][1], datasetDict[segType][i][2], datasetDict[segType][i][3] ]).reshape((len(datasetDict[segType][i][0]), 4, 7))
                # 1. normalize
                x_test = model['arterial']['scaler'].transform(dataset.reshape(1367, -1))
                x_test = x_test.reshape(1367, 4, -1)
                # 2. fit into model
                y_pred = model['arterial']['model'].predict_proba(x_test)[:, 1]
                y_pred_adjusted = (((1/16000)*y_pred*100)/(1 - y_pred + (1/16000)*y_pred)).tolist()
                # segment_predict = pd.DataFrame({'allId1': , 'lstmCrashRisk': ((1/4000)*y_pred*100)/(1 - y_pred + (1/4000)*y_pred)})
                # print(len(y_pred_adjusted))
                datasetDict[segType][i].pop(0)
                return y_pred_adjusted
            elif segType == 'intersection':
                dataset = np.asarray([ datasetDict[segType][i][0], datasetDict[segType][i][1], datasetDict[segType][i][2], datasetDict[segType][i][3] ]).reshape((len(datasetDict[segType][i][0]), 4, 10))
                # 1. normalize
                x_test = model['intersection']['scaler'].transform(dataset.reshape(514, -1))
                x_test = x_test.reshape(514, 4, -1)
                # 2. fit into model
                y_pred = model['intersection']['model'].predict_proba(x_test)[:, 1]
                y_pred_adjusted = (((1/16000)*y_pred*100)/(1 - y_pred + (1/16000)*y_pred)).tolist()
                # segment_predict = pd.DataFrame({'allId1': , 'lstmCrashRisk': ((1/4000)*y_pred*100)/(1 - y_pred + (1/4000)*y_pred)})
                # print(len(y_pred_adjusted))
                datasetDict[segType][i].pop(0)
                return y_pred_adjusted
        else:
            continue
    return None
def lstmDatasetPreprocessing(df, segType):
    if segType == 'arterial':
        df = df[['avgSpd','downTotalVolume','humidity','stdSpd','upTotalVolume','visibility','weatherType']]
        arterialDataset = []
        for index, rows in df.iterrows():
            arterialDataset.append([rows.avgSpd, rows.downTotalVolume, rows.humidity, rows.stdSpd, rows.upTotalVolume, rows.visibility, rows.weatherType])
        return arterialDataset
    if segType == 'intersection':
        df = df[['majorAvgSpd', 'majorStdSpd', 'visibility', 'humidity', 'precipitation', 'majorVolume', 'majorAoy', 'majorGreenRatio', 'minorVolume', 'minorGreenRatio']]
        intersectionDataset = []
        for index, rows in df.iterrows():
            intersectionDataset.append([rows.majorAvgSpd, rows.majorStdSpd, rows.visibility, rows.humidity, rows.precipitation, rows.majorVolume, rows.majorAoy, rows.majorGreenRatio, rows.minorVolume, rows.minorGreenRatio])
        return intersectionDataset

def dataFiltering(data_query, base_map_segment_df, base_map_intersection_df, backup, datasetDict, times, model):
    dotDataDict = data_query.action_data_query({'c2c':5, 'TMC':5, 'weather':0}, False)
    if 'weather' not in dotDataDict:
        return 0
    elif 'c2c' not in dotDataDict:
        return 0
    elif 'TMC' not in dotDataDict:
        return 0

    c2c_groupby_df, approach_groupby_df = dotDataPreprocessing(dotDataDict)
    if c2c_groupby_df is None or approach_groupby_df is None:
        return 0
    preprocessing_base_map_segment_df = baseMapPreprocessing(base_map_segment_df.copy(), 'segment', dotDataDict['c2c'][0], dotDataDict['weather'], c2c_groupby_df, approach_groupby_df)
    freeway_basic_segment_df = freewayBasicSegment(preprocessing_base_map_segment_df.copy().loc[preprocessing_base_map_segment_df['Seg_Type'] == 'Frw_Basic_Segment'], dotDataDict['c2c'][0], backup)
    freeway_weave_segment_df = freewayWeavingSegment(preprocessing_base_map_segment_df.copy().loc[preprocessing_base_map_segment_df['Seg_Type'] == 'Frw_Weav_Segment'], dotDataDict['c2c'][0], backup)
    freeway_ramp_segment_df = freewayRampSegment(preprocessing_base_map_segment_df.copy().loc[preprocessing_base_map_segment_df['Seg_Type'].str.contains('Rmp')], dotDataDict['c2c'][0], backup)
    arterial_segment_df = arterialSegment(preprocessing_base_map_segment_df.copy().loc[preprocessing_base_map_segment_df['Seg_Type'] == 'Arterial_Segment'], dotDataDict['c2c'][0], backup)

    preprocessing_intersection_df = baseMapPreprocessing(base_map_intersection_df.copy(), 'intersection', dotDataDict['c2c'][0], dotDataDict['weather'], c2c_groupby_df, approach_groupby_df)
    intersection_df = intersection(preprocessing_intersection_df.copy(), dotDataDict['c2c'][0], backup)

    datasetDict['arterial'][times % 5].append(lstmDatasetPreprocessing(arterial_segment_df.copy(), 'arterial'))
    predArterialList = lstmPredicting(datasetDict, 'arterial', model)
    if predArterialList != None:
        arterial_segment_df['crashRiskLSTM'] = predArterialList
        arterial_segment_df['crashRisk'] = (arterial_segment_df['crashRiskLSTM'] + arterial_segment_df['crashRisk'])/2
        var_ = np.exp( (-5.8556) +
                    0.0512 * arterial_segment_df['avgSpd'])
        arterial_segment_df['severeCrashRisk'] = (var_/ (1 + var_)) * arterial_segment_df['crashRisk']

        # arterial_segment_df.drop(columns=['crashRiskLSTM']).to_csv('arterial' + str(times%5) + '.csv')

    datasetDict['intersection'][times % 5].append(lstmDatasetPreprocessing(intersection_df.copy(), 'intersection'))
    predIntersectionList = lstmPredicting(datasetDict, 'intersection', model)
    if predIntersectionList != None:
        intersection_df['crashRiskLSTM'] = predIntersectionList
        intersection_df['crashRisk'] = (intersection_df['crashRiskLSTM'] + intersection_df['crashRisk'])/2

        var_ = np.exp( (-2.3455) + (-1.1262) * intersection_df['minorGreenRatio'])
        intersection_df['severeCrashRisk'] = (var_/ (1 + var_)) * intersection_df['crashRisk']

        # intersection_df.drop(columns=['crashRiskLSTM']).to_csv('intersection' + str(times%5) + '.csv')
    # freeway_basic_segment_df.to_csv("D:\\save\\" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_basic.csv")
    # freeway_weave_segment_df.to_csv("D:\\save\\" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_weaving.csv")
    # freeway_ramp_segment_df.to_csv("D:\\save\\" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_ramp.csv")
    # arterial_segment_df.to_csv("D:\\save\\" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_arterial.csv")
    # intersection_df.to_csv("D:\\save\\" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_intersection.csv")
    try:
        upload2DB(freeway_basic_segment_df, 'http://45.55.42.130/basic/many', 90)
        upload2DB(freeway_ramp_segment_df, 'http://45.55.42.130/ramp/many', 90)
        upload2DB(freeway_weave_segment_df, 'http://45.55.42.130/weaving/many', 31)
        upload2DB(arterial_segment_df, 'http://45.55.42.130/arterial/many', 64)
        upload2DB(intersection_df, 'http://45.55.42.130/intersection/many', 32)

        copy_basic_df = freeway_basic_segment_df[['allId1', 'crashRisk', 'severeCrashRisk']].rename(index=str, columns={'allId1':'id'})
        copy_basic_df['roadwayType'] = 'basic'
        copy_ramp_df = freeway_ramp_segment_df[['allId1', 'crashRisk', 'severeCrashRisk']].rename(index=str, columns={'allId1':'id'})
        copy_ramp_df['roadwayType'] = 'ramp'
        copy_weave_df = freeway_weave_segment_df[['allId1', 'crashRisk', 'severeCrashRisk']].rename(index=str, columns={'allId1':'id'})
        copy_weave_df['roadwayType'] = 'weaving'
        copy_arterial_df = arterial_segment_df[['allId1', 'crashRisk', 'severeCrashRisk']].rename(index=str, columns={'allId1':'id'})
        copy_arterial_df['roadwayType'] = 'arterial'
        copy_intersection_df = intersection_df[['intersection', 'crashRisk', 'severeCrashRisk']].rename(index=str, columns={'intersection':'id'})
        copy_intersection_df['roadwayType'] = 'intersection'


        upload2DB(copy_basic_df[copy_basic_df['crashRisk'] > 0.126 ], 'http://165.227.71.154/realtime/many', 100)
        upload2DB(copy_ramp_df[copy_ramp_df['crashRisk'] > 0.002], 'http://165.227.71.154/realtime/many', 100)
        upload2DB(copy_weave_df[copy_weave_df['crashRisk'] > 0.067], 'http://165.227.71.154/realtime/many', 100)
        upload2DB(copy_arterial_df[copy_arterial_df['crashRisk'] > 0.01], 'http://165.227.71.154/realtime/many', 100)
        upload2DB(copy_intersection_df[copy_intersection_df['crashRisk'] > 0.026], 'http://165.227.71.154/realtime/many', 100)
    except:
        print('Roadway Filter:', time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime(time.time())) ,' upload data error.')



def roadwayFilter():
    os.environ["CUDA_VISIBLE_DEVICES"]="7"
    scaler_segment = joblib.load('model/scaler_segment.save')
    model_segment = load_model('model/final_segment_lstm.h5')

    scaler_intersection = joblib.load('model/scaler_intersection.save')
    model_intersection = load_model('model/final_intersection_lstm.h5')

    model = {'arterial':{'model':model_segment, 'scaler': scaler_segment}, 'intersection':{'model':model_intersection, 'scaler':scaler_intersection}}

    base_map_segment_df = pd.read_csv('basemap/Final_map_segment_0516.csv')
    base_map_intersection_df = pd.read_csv('basemap/Final_map_intersection_0516.csv')
    data_query = query_action_data()
    backup = readBackupFile()
    datasetDict = {'arterial':[[],[],[],[],[]], 'intersection':[[],[],[],[],[]]}
    times = 0
    while True:
        start_time = time.time()
        if dataFiltering(data_query, base_map_segment_df, base_map_intersection_df, backup, datasetDict, times, model) == 0:
            time.sleep(1)
            continue
        else:
            times = times + 1
            if times % 5 == 0:
                times = 0
            if 60 - (time.time() - start_time) > 0:
                print('Roadway Filter:', time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime(time.time())) ,' Whole processing costs:', int(time.time() - start_time), 'seconds.')
                time.sleep( 60 - int(time.time() - start_time))
            else:
                print('Roadway Filter:', time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime(time.time())) ,' Whole processing costs:', int(time.time() - start_time), 'seconds.')
                continue
