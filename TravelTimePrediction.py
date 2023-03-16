"""
Created on Fri Jan 17 20:01:56 2020

@author: ji758507
"""
import pandas as pd
import numpy as np
import matplotlib as plt
from datetime import timedelta


# ==========================================================================================================
# get bluetoad speed travel time data
# ==========================================================================================================
def bluetoad_speed_travel_time(BlueToad_Full_Data, time_window, Full_PairID_list):
    ts = pd.date_range(start='2018-07-01 00:00:00', end='2019-01-01 00:00:00', freq='T')  # Generate time series
    ts = ts.to_frame().reset_index(drop=True)
    ts.rename(columns={0: 'Time'}, inplace=True)

    ts_r = ts.reindex(np.repeat(ts.index.values, time_window), method='ffill')
    ts_r['Time_interval'] = ts_r['Time']
    ts_r['Time_interval'] -= pd.TimedeltaIndex(ts_r.groupby(level=0).cumcount(), unit='m')
    BlueToad_Full_Data['Date_Time'] = pd.to_datetime(BlueToad_Full_Data['Date_Time'])

    for i in Full_PairID_list:
        BT_tem = BlueToad_Full_Data[BlueToad_Full_Data['PairID'] == i].sort_values(by=['Date_Time'])
        speed = ts_r.join(BT_tem.set_index('Date_Time'), on='Time_interval')
        speed.drop(['Time_interval', 'PairID', 'Day of week'], axis=1, inplace=True)
        speed = speed.groupby(by=['Time'], as_index=False).agg({"Speed (mph)": ['mean', 'std', 'count'], "Travel time (s)": ['mean', 'std']})
        speed.columns = ['Time', 'Avg_Speed_{}'.format(i), 'Std_Speed_{}'.format(i), 'Count_{}'.format(i), 'Avg_Travel_Time_{}'.format(i), 'Std_Travel_Time_{}'.format(i)]

        ts = ts.join(speed.set_index('Time'), on='Time')
    ts.to_csv('./Data/Bluetoad/Speed_Travel_Time_Full_Data.csv', sep=',')  # Export the data

    return ts


# get the full list of bluetoad pair ids
# Call the function to calculate the (5-min level) avg_speed and std_speed for every PairID and timestamp
# Speed_Travel_Time_Full_Data = bluetoad_speed_travel_time(BlueToad_Full_Data, 5, Full_PairID_list)
# Speed_Travel_Time_Full_Data.iloc[:, Speed_Travel_Time_Full_Data.columns.str.contains('Count')].describe()

# data visulization
# test = pd.concat([Speed_Travel_Time_Full_Data.Time, Speed_Travel_Time_Full_Data.iloc[:, Speed_Travel_Time_Full_Data.columns.str.contains('Count')]], axis=1)
# test.dtypes
#
# for i in Full_PairID_list:
#     bt_pair_data = test.loc[:, ['Count_{}'.format(i)]]
#     bt_pair_data = bt_pair_data.set_index(test.Time)
#     weekly = bt_pair_data.resample('W').sum() / 5
#     weekly.plot()
#     plt.show()

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)

# ==========================================================================================================
# Event Data (data is incomplete, not included in the study )
# ==========================================================================================================
# event_data_col_name = pd.read_excel('Data/Event Data/EVENT_ColumnName.xlsx')
# event_data = pd.read_csv('Data/Event Data/EM_EVENT.csv', names=event_data_col_name['Col_Name'].tolist())
# event_data['Latitude'] = event_data['POINT_LATITUDE']/1000000
# event_data['Longitude'] = event_data['POINT_LONGITUDE']/1000000
# event_data['CREATED_DATE'] = pd.to_datetime(event_data['CREATED_DATE'])
# event_data = event_data[(event_data['CREATED_DATE'] >= '2018-07-01 00:00:00') & (event_data['CREATED_DATE'] <= '2019-01-01 00:00:00')].reset_index(drop=True)
# event_data = event_data.sort_values(by='EVENT_ID').reset_index(drop=True)
# event_data.to_csv('Data/Event Data/Event_Data_Processed.csv', sep=',')
#
# event_data.groupby('EVENTSTATUS_ID').size()
# event_data.groupby('EVENTTYPE_ID').size()


# ==========================================================================================================
# Integrate all the data sources and convert to 5-min data
# ==========================================================================================================
def convert_cycle_to_minutes(atspm_data, SR436_Selected_Segments, time_steps, weather_data, Speed_Travel_Time_Full_Data, prediction_horizon):
    """

    :param raw_data:
    :param intersection_approaches_dict:
    :return:
    """
    ts = pd.date_range(start='2018-07-01 00:00:00', end='2019-01-01 00:00:00', freq='T') # Generate time series
    # temp = ts.to_frame().reset_index(drop=True)
    # temp.rename(columns={0: 'Time'}, inplace=True)
    y_temp = ts.to_frame().reset_index(drop=True)
    y_temp.rename(columns={0: 'Time'}, inplace=True)

    for inner_outer in range(0, 2):

        segments = SR436_Selected_Segments[SR436_Selected_Segments['inner_outer'] == inner_outer].sort_values(by=['ES_seq']).reset_index(drop=True)

        for i in range(len(segments)):

            # generate reference table
            ts_r = ts.to_frame().reset_index(drop=True)
            ts_r.rename(columns={0: 'Time'}, inplace=True)
            ts_r = ts_r.reindex(np.repeat(ts_r.index.values, 5), method='ffill')
            ts_r['Time_interval'] = ts_r['Time']
            ts_r['Time_interval'] -= pd.TimedeltaIndex(ts_r.groupby(level=0).cumcount(), unit='m')

            ts1 = ts.to_frame().reset_index(drop=True)
            ts1.rename(columns={0: 'Time'}, inplace=True)

            pair_id = segments.loc[i, 'Pair_ID']

            for j in list(['Up', 'Down']):
                signal_id = int(segments.loc[i, '{}_signal'.format(j)][5:9])
                direction = segments.loc[i, '{}_Dir'.format(j)][0:1]

                # extract the data cycle data for given signal_id and direction
                temp_data = atspm_data[(atspm_data['signal_id'] == signal_id) & (atspm_data['movement'].str.contains(direction))].reset_index(drop=True)

                # generate the ceiling and floor of every cycle
                temp_data['cycle_floor'] = temp_data['BOC'].dt.floor("min")
                temp_data['cycle_ceil'] = temp_data['EOC'].dt.ceil("min")
                temp_data['num_minutes'] = ((temp_data['cycle_ceil'] - temp_data['cycle_floor']).dt.total_seconds()/60).astype(int)

                # repeat rows based on num_minutes
                temp_data = temp_data.loc[temp_data.index.repeat(temp_data['num_minutes'])]

                # group by index with transform for date ranges (minute), get splited minute floor
                # temp_data['minute_floor'] = (temp_data.groupby(level=0)['cycle_floor'].transform(lambda x: pd.date_range(start=x.iat[0], periods=len(x), freq='T')))
                temp_data['minute_floor'] = temp_data['cycle_floor']
                temp_data['minute_floor'] += pd.TimedeltaIndex(temp_data.groupby(level=0).cumcount(), unit='m')

                # identify the boundary between cycles, and split to corresponding minutes
                temp_data = temp_data.reset_index(drop=True)
                temp_data['cycle_split1'] = (temp_data['BOC'] - temp_data['minute_floor']).dt.total_seconds()
                temp_data['cycle_split2'] = (temp_data['EOC'] - temp_data['minute_floor']).dt.total_seconds()

                # consider different conditions to calculate the split duration
                temp_data.loc[(temp_data['cycle_split1'] >= 0) & (temp_data['num_minutes'] > 1), 'split_duration'] = 60 - temp_data['cycle_split1']
                temp_data.loc[(temp_data['cycle_split2'] <= 60) & (temp_data['num_minutes'] > 1), 'split_duration'] = temp_data['cycle_split2']
                temp_data.loc[temp_data['num_minutes'] == 1, 'split_duration'] = temp_data['cycle_len']
                temp_data['split_duration'] = temp_data['split_duration'].fillna(60)

                # calculate the weight of every minute
                temp_data['minute_weight'] = temp_data['split_duration']/60
                temp_data = temp_data[['signal_id', 'movement', 'cycle_num', 'cycle_len', 'green_ratio', 'BOC', 'EOC', 'BOR', 'EOR', 'BOG', 'EOG',
                                       'green_time', 'yellow_time', 'red_time', 'cycle_volume', 'aog', 'aoy', 'aor', 'pog', 'poy', 'por', 'aogr', 'aoyr', 'aorr', 'platoon_ratio', 'avg_occupancy_green',
                                       'std_occupancy_green', 'avg_headway_green', 'std_headway_green', 'avg_occupancy_red', 'std_occupancy_red',
                                       'avg_headway_red', 'std_headway_red', 'max_queue_length', 'shock_wave_area', 'queuing_shockwave_spd', 'minute_floor', 'minute_weight']]

                # multiply the variables with the weight column
                temp_data[['cycle_len', 'green_ratio', 'green_time', 'yellow_time', 'red_time', 'cycle_volume', 'aog', 'aoy', 'aor', 'pog', 'poy', 'por', 'aogr', 'aoyr', 'aorr', 'platoon_ratio', 'avg_occupancy_green',
                           'std_occupancy_green', 'avg_headway_green', 'std_headway_green', 'avg_occupancy_red', 'std_occupancy_red',
                           'avg_headway_red', 'std_headway_red', 'max_queue_length', 'shock_wave_area', 'queuing_shockwave_spd']] \
                    = temp_data[['cycle_len', 'green_ratio', 'green_time', 'yellow_time', 'red_time', 'cycle_volume', 'aog', 'aoy', 'aor', 'pog', 'poy', 'por', 'aogr', 'aoyr', 'aorr', 'platoon_ratio', 'avg_occupancy_green',
                           'std_occupancy_green', 'avg_headway_green', 'std_headway_green', 'avg_occupancy_red', 'std_occupancy_red',
                           'avg_headway_red', 'std_headway_red', 'max_queue_length', 'shock_wave_area', 'queuing_shockwave_spd']].multiply(temp_data["minute_weight"], axis="index")

                # groupby the mintue_floor
                temp_data = temp_data.groupby(by=['minute_floor'], as_index=False)[['cycle_len', 'green_ratio', 'green_time', 'yellow_time', 'red_time', 'cycle_volume', 'aog', 'aoy', 'aor', 'pog', 'poy', 'por', 'aogr', 'aoyr', 'aorr', 'platoon_ratio', 'avg_occupancy_green',
                           'std_occupancy_green', 'avg_headway_green', 'std_headway_green', 'avg_occupancy_red', 'std_occupancy_red',
                           'avg_headway_red', 'std_headway_red', 'max_queue_length', 'shock_wave_area', 'queuing_shockwave_spd']].sum().reset_index(drop=True)

                # join temp_data to the reference table
                ts_join = ts_r.join(temp_data.set_index('minute_floor'), on='Time_interval')

                # groupby five minutes and then calculate the mean value for all the variables
                ts_join = ts_join.groupby(by=['Time'], as_index=False)[['cycle_len', 'green_ratio', 'green_time', 'yellow_time', 'red_time', 'cycle_volume', 'aog', 'aoy', 'aor', 'pog', 'poy', 'por', 'aogr', 'aoyr', 'aorr', 'platoon_ratio', 'avg_occupancy_green',
                           'std_occupancy_green', 'avg_headway_green', 'std_headway_green', 'avg_occupancy_red', 'std_occupancy_red',
                           'avg_headway_red', 'std_headway_red', 'max_queue_length', 'shock_wave_area', 'queuing_shockwave_spd']].mean().reset_index(drop=True)

                # add suffix to the variables to distinguish up and down
                ts_join = ts_join.add_suffix('_{}'.format(j))

                ts1 = ts1.join(ts_join.set_index('Time_{}'.format(j)), on='Time')

            # add weather variables
            ts1 = ts1.join(weather_data.set_index('index'), on='Time')

            # add bluetoad variables
            bt_pair_data = pd.concat([Speed_Travel_Time_Full_Data.Time, Speed_Travel_Time_Full_Data.iloc[:, Speed_Travel_Time_Full_Data.columns.str.contains(pair_id)]], axis=1)
            bt_pair_data.columns = ['Time', 'avg_speed', 'std_speed', 'count', 'avg_travel_time', 'std_travel_time']
            ts1 = ts1.join(bt_pair_data.set_index('Time'), on='Time')

            # add suffix to the variables
            ts1_1 = ts1.add_suffix('_{}_{}_1'.format(inner_outer, i))
            ts1_1.rename(columns={'Time_{}_{}_1'.format(inner_outer, i): 'Time'}, inplace=True)

            # add variables for other time steps (2, 3, 4, 5, 6)
            for k in range(1, time_steps):
                ts_temp = ts.to_frame().reset_index(drop=True)
                ts_temp.rename(columns={0: 'Time'}, inplace=True)

                # every time slice starts from the last 5 minutes
                # ts_temp['Time_slice'] = ts_temp['Time'] - timedelta(minutes=5 * k)

                # every time slice starts from the last minutes
                ts_temp['Time_slice'] = ts_temp['Time'] - timedelta(minutes=k)

                # join data with the reference table
                ts_temp = ts_temp.join(ts1.set_index('Time'), on='Time_slice')
                ts_temp = ts_temp.add_suffix('_{}_{}_{}'.format(inner_outer, i, k + 1))
                ts_temp = ts_temp.drop(['Time_slice_{}_{}_{}'.format(inner_outer, i, k + 1)], axis=1)

                ts1_1 = ts1_1.join(ts_temp.set_index('Time_{}_{}_{}'.format(inner_outer, i, k + 1)), on='Time')

            # data cleaning and fillna
            ts1_1 = ts1_1.fillna(method='ffill', axis=0)
            ts1_1 = ts1_1.fillna(method='backfill', axis=0)

            # delete data from 7 months
            # ts_join_1['month'] = ts_join_1['Time'].apply(lambda x: '{}{}'.format(x.year, x.month))
            # ts_join_1 = ts_join_1[~ts_join_1['month'].isin(['20178', '201710', '201711', '201712', '20182', '20185', '20186'])]
            # ts_join_1 = ts_join_1.drop(['month'], axis=1)
            y_temp['y_{}_{}'.format(inner_outer, i)] = ts1_1['avg_travel_time_{}_{}_1'.format(inner_outer, i)].shift(-1*prediction_horizon)

            ts1_1 = ts1_1[~ts1_1['Time'].map(lambda x: 100 * x.year + x.month).isin([201807, 201808, 201809, 201812, 201901])].reset_index(drop=True)
            ts1_1 = ts1_1.iloc[time_steps:-prediction_horizon, :]
            # ts_join_1.to_csv('./Data/split_data/approach_data_{}_{}_{}_{}_{}.csv'.format(group, inner_outer, i, signal_id, direction), sep=',')
            ts1_1.to_csv('Data/split_data/30_min/segment_data_{}_{}_{}.csv'.format(inner_outer, i, pair_id), sep=',')

            print(inner_outer, i, pair_id, len(ts1_1))

    y_temp = y_temp[~y_temp['Time'].map(lambda x: 100 * x.year + x.month).isin([201807, 201808, 201809, 201812, 201901])].reset_index(drop=True)
    y_temp = y_temp.iloc[time_steps:-prediction_horizon, :]
    y_temp.to_csv('Data/split_data/30_min/y_segment_travel_time.csv', sep=',')
    print(len(y_temp))


Speed_Travel_Time_Full_Data = pd.read_csv('./Data/Bluetoad/Speed_Travel_Time_Full_Data.csv')
Speed_Travel_Time_Full_Data = Speed_Travel_Time_Full_Data.drop(columns='Unnamed: 0')
Speed_Travel_Time_Full_Data['Time'] = pd.to_datetime(Speed_Travel_Time_Full_Data['Time'])

SR436_Selected_Segments = pd.read_csv('Data/SR436_Selected_Segments.csv')
intersection_approaches_dict = pd.read_csv('J:\ATSPM_CNN_LSTM\Data\intersection_approaches_dict.csv')
SR436_Selected_Segments = pd.merge(SR436_Selected_Segments, intersection_approaches_dict[['Seg_ID', 'ES_seq', 'inner_outer']], how='left', left_on=['Seg_ID'], right_on=['Seg_ID'])

# Full_PairID_list = SR436_Selected_Segments['Pair_ID'].tolist()
# ==========================================================================================================
# ATSPM Data
# ==========================================================================================================
atspm_data = pd.read_csv('Data/raw_data_SR436_Selected.csv')
atspm_data.iloc[:, 4:10] = atspm_data.iloc[:, 4:10].apply(pd.to_datetime, errors='coerce')


# ==========================================================================================================
# Weather Data（station: 12815; 12854）
# ==========================================================================================================
weather_data = pd.read_csv(r'J:\ATSPM_CNN_LSTM\Data\Weather Data\weather_data_processed.csv')
weather_data = weather_data.drop(columns='Unnamed: 0')
weather_data['Visibility'] = weather_data.iloc[:, weather_data.columns.str.contains('Visibility')].mean(axis=1)
weather_data['WeatherTypes'] = weather_data.iloc[:, weather_data.columns.str.contains('WeatherTypes')].max(axis=1)
weather_data['Humidity'] = weather_data.iloc[:, weather_data.columns.str.contains('Humidity')].mean(axis=1)
weather_data['Precipitation'] = weather_data.iloc[:, weather_data.columns.str.contains('Precipitation')].mean(axis=1)
weather_data = weather_data.loc[:, ~weather_data.columns.str.contains('_')]
weather_data['index'] = pd.to_datetime(weather_data['index'])


# prediction_horizon = 5
if __name__ == '__main__':
    convert_cycle_to_minutes(atspm_data, SR436_Selected_Segments, 30, weather_data, Speed_Travel_Time_Full_Data, 30)
