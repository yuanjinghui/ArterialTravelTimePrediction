import json
import pandas as pd
import time
import datetime
import math
import numpy as np
from datetime import timedelta
from pandas.io.json import json_normalize
from time import strftime, gmtime
from requests import request
import requests

def realTimeQuery(ip='http://165.227.71.154/realtime/getDataInDays/7', show_cost_time=False):
    try:
        start_time = time.time()
        response=request(url=ip, timeout=60, method='get')
        data = response.json()
        if show_cost_time:
            print('query data from real time costs', int(time.time() - start_time), 'seconds.')
    except:
        print('real time query error')
        return None

    df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')

    if 'id' in df:
        df['createdDate'] = pd.to_datetime( df['createdDate'] ) - datetime.timedelta(hours=4)
        return df.drop(columns=['__v', '_id'])
    else:
        return None

def upload2DB(dataList, Ip, numberOfSubmit):
    submitList = []
    for i in range(len(dataList)):
        submitList.append(dataList[i])
        if len(submitList) >= numberOfSubmit:
            r = requests.post(Ip, json=submitList)
            if  r.status_code != 200:
                print('Upload file to', Ip, 'error, error code number:', r.status_code)
            # print('Upload', numberOfSubmit, 'files to', Ip, 'at', datetime.datetime.now(), r.status_code)
            submitList.clear()
    if len(submitList) != 0:
        r = requests.post(Ip, json=submitList)
        if  r.status_code != 200:
            print('Upload file to', Ip, 'error, error code number:', r.status_code)
        # print('Upload', len(submitList), 'files to', Ip, 'at', datetime.datetime.now(), r.status_code)
        submitList.clear()

def generate_high_risk_freq(base_map_segment_df, base_map_intersection_df):
    df = realTimeQuery(ip='http://165.227.71.154/realtime/getDataInDays/7', show_cost_time=True)
    if df is None:
        return None
    df['halfHour'] = df['createdDate'].apply(lambda x: math.ceil((x.hour * 60 + x.minute) / 30))
    realtime_high_risk_freq = df.groupby(by=['id', 'roadwayType', 'halfHour'], as_index=False).size().reset_index(name='highRiskFreq')

    segment_df = base_map_segment_df[['All_ID1', 'Seg_Type']].copy()
    segment_df.loc[segment_df['Seg_Type'] == 'Frw_Basic_Segment', 'roadwayType'] = 'basic'
    segment_df.loc[segment_df['Seg_Type'] == 'Frw_Weav_Segment', 'roadwayType'] = 'weaving'
    segment_df.loc[segment_df['Seg_Type'].str.contains('Rmp'), 'roadwayType'] = 'ramp'
    segment_df.loc[segment_df['Seg_Type'] == 'Arterial_Segment', 'roadwayType'] = 'arterial'
    segment_df = segment_df[['All_ID1', 'roadwayType']]
    segment_df.rename(columns={'All_ID1': 'id'}, inplace=True)
    segment_df['id'] = segment_df.id.astype(str)

    intersection_df = pd.DataFrame(base_map_intersection_df['Intersection'].copy())
    intersection_df['roadwayType'] = 'intersection'
    intersection_df.rename(columns={'Intersection': 'id'}, inplace=True)

    roadway_df = pd.concat([segment_df, intersection_df]).sort_values('roadwayType').reset_index(drop=True)
    roadway_df = roadway_df.reset_index(drop=False).rename(columns={'index': 'uniqueId'})

    data = roadway_df.iloc[:, :].copy()
    data = data.set_index('uniqueId')
    # create a empty table
    for i in range(0, 48):
        data['{}'.format(i)] = 0
        data['{}'.format(i)] = data['{}'.format(i)].astype(int)

    for key in range(0, 48):
        temp = realtime_high_risk_freq[realtime_high_risk_freq['halfHour'] == (key + 1)][['id', 'roadwayType', 'highRiskFreq']]
        temp.rename(columns={'highRiskFreq': '{}'.format(key)}, inplace=True)

        if len(temp) > 0:
            temp = pd.merge(temp, roadway_df, how='left', left_on=['id', 'roadwayType'], right_on=['id', 'roadwayType'])
            temp = temp.set_index('uniqueId')
            temp = temp.drop(columns=['id', 'roadwayType'])
            data.update(temp)

    # AM: 6-9, PM: 17-20, DayNonPeak: 10-16, NightNonPeak: 0-5 and 21-23
    data['amPeak'] = data.loc[:, ['13', '14', '15', '16', '17', '18', '19', '20']].sum(axis=1)
    data['pmPeak'] = data.loc[:, ['35', '36', '37', '38', '39', '40', '41', '42']].sum(axis=1)
    data['dayNonPeak'] = data.loc[:, ['21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34']].sum(axis=1)
    data['nightNonPeak'] = data.loc[:, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '43', '44', '45', '46', '47', '48']].sum(axis=1)


    dataList = []
    for j in range(2, 54):
        dataDict = {}
        name = data.columns[j]
        temp1 = data.iloc[:, np.r_[0:2, j]]
        temp1 = temp1[temp1[name] > 0].reset_index(drop=True)
        temp1.rename(columns={name: 'highCrashRiskFreq'}, inplace=True)
        temp2 = temp1.T
        dataDict['segments'] = []
        for element in temp2:
            subDataDict = {}
            subDataDict['id'] = temp2[element][0]
            subDataDict['roadwayType'] = temp2[element][1]
            subDataDict['highCrashRiskFreq'] = temp2[element][2]
            dataDict['segments'].append(subDataDict)
        dataDict['time'] = name
        dataList.append(dataDict)
    return dataList


if __name__ == '__main__':
    base_map_segment_df = pd.read_csv('Final_map_segment_0516.csv', index_col=0)
    base_map_intersection_df = pd.read_csv('Final_map_intersection_0516.csv', index_col=0)
    day = -1

    while True:
        sysDay = int(strftime('%m-%d-%Y %H:%M:%S', gmtime(time.time() - 3600 * 4))[3:5])
        if sysDay != day:
            dataList = generate_high_risk_freq(base_map_segment_df, base_map_intersection_df)
            if dataList is not None:
                upload2DB(dataList, 'http://165.227.71.154/highRiskFreq/many', 1)
                print('Updating finish.')
                day = sysDay
            else:
                time.sleep(1)
                continue
        else:
            time.sleep(3600)



realtime_high_risk_freq.highRiskFreq.describe()
realtime_high_risk_freq.highRiskFreq.quantile(0.8)