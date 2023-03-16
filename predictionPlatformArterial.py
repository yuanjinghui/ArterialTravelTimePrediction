"""
Created on Mon Apr 27 20:01:56 2020

@author: ji758507
"""
import pandas as pd
from datetime import timedelta
import datetime
import numpy as np
import time
import requests
import os
from time import strftime, gmtime
from keras.models import load_model
from sklearn.externals import joblib
import h5py


def load_data_for_convLSTM(data_file):
    with h5py.File(data_file, 'r') as f:
        print(list(f.keys()))
        x_train = f['x_train'][:]
        y_train = f['y_train'][:]
        x_test = f['x_test'][:]
        y_test = f['y_test'][:]
        f.close()
    return x_train, y_train, x_test, y_test


def x_reshape(x_train):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], -1)
    a = x_train.shape[1]
    b = x_train.shape[2]
    x_train = np.moveaxis(x_train, -1, 1)
    x_train = x_train.reshape(-1, a, b)
    return x_train


def label_reshape_gru(y_pred, num_samples):
    y_pred = y_pred.reshape(num_samples, -1)
    scaler_filename = "scaler.save"
    scaler = joblib.load(scaler_filename)
    y_pred_r = scaler.inverse_transform(y_pred)
    # 2d array to 1d array
    y_pred_r = y_pred_r.ravel()
    return y_pred_r


def upload2DB(df, Ip):

    r = requests.post(Ip, json= df)
    if r.status_code != 200:
        print("error occur, error code is ", r.status_code)
    else:
        print("success to upload", r.status_code)


def getPrediction(input_index, oneWeekInput, model_5, predictionHorizon, test_time):
    x_current = oneWeekInput[input_index: input_index + 1, :, :, :, :]
    x_current.shape
    num_samples = x_current.shape[0]
    x_current_r = x_reshape(x_current)

    y_pred = model_5.predict(x_current_r)
    print(x_current_r.shape, y_pred.shape)

    y_pred_r = label_reshape_gru(y_pred, num_samples)
    print(y_pred_r.shape)

    y_pred_r = y_pred_r.reshape(num_samples, 4, 2)

    time = pd.DataFrame(test_time.iloc[input_index: input_index + 1, :])
    for inner_outer in range(2):
        for i in range(4):
            prediction = pd.DataFrame({'Time': time.time, 'y_{}_{}'.format(inner_outer, i): y_pred_r[:, i, inner_outer]})
            time = time.join(prediction.set_index('Time'), on='time')

    time = time.reset_index(drop=True)
    time['predictionHorizon'] = predictionHorizon
    return time


def read_historical(start, end, folder_path):
    files = os.listdir(folder_path)
    files = [f for f in files if 'segment_data' in f]

    historical = {}
    for i in files:
        segment_path = os.path.join(folder_path, i)
        inner_outer = int(i[13])
        segment_id = int(i[15])
        historical_data = pd.read_csv(segment_path, index_col=0)
        historical_data['Time'] = pd.to_datetime(historical_data['Time'])
        historical_data = historical_data.loc[(historical_data['Time'] >= datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S'))
                                              & (historical_data['Time'] <= datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S'))].reset_index(drop=True)

        historical['data_{}_{}'.format(inner_outer, segment_id)] = historical_data
        del historical_data

    return historical


def gen_historical_data(start, end, folder_path, input_index, start_time, data):
    files = os.listdir(folder_path)
    files = [f for f in files if 'segment_data' in f]

    ts_temp = pd.date_range(start=start_time - timedelta(minutes=31), end=start_time - timedelta(minutes=1), freq='T')[::-1]  # Generate time series
    ts_temp = [t.strftime("%Y-%m-%d %H:%M:%S") for t in ts_temp]
    historical_volume = pd.DataFrame(ts_temp).reset_index(drop=True)
    historical_volume.rename(columns={0: 'Time'}, inplace=True)

    historical_travel_time = pd.DataFrame(ts_temp).reset_index(drop=True)
    historical_travel_time.rename(columns={0: 'Time'}, inplace=True)

    for i in files:
        inner_outer = int(i[13])
        segment_id = int(i[15])
        historical_data = data['data_{}_{}'.format(inner_outer, segment_id)]
        historical_data['Time'] = pd.to_datetime(historical_data['Time'])
        historical_data = historical_data.loc[(historical_data['Time'] >= datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S'))
                                              & (historical_data['Time'] <= datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S'))].reset_index(drop=True)

        tem = historical_data.loc[:, historical_data.columns.str.contains('|'.join(['Time', 'cycle_volume_Down', 'avg_travel_time']))]
        # tem.columns = [str(col) + '_' for col in tem.columns]
        # tem = tem.rename(columns={'Time_': 'Time'})
        # tem = tem.loc[:, tem.columns.str.contains('|'.join(['Time', '_{}_{}_5_'.format(inner_outer, segment_id), '_{}_{}_10_'.format(inner_outer, segment_id),
        #                                                     '_{}_{}_15_'.format(inner_outer, segment_id), '_{}_{}_20_'.format(inner_outer, segment_id),
        #                                                     '_{}_{}_25_'.format(inner_outer, segment_id), '_{}_{}_30_'.format(inner_outer, segment_id)]))]

        volume = pd.DataFrame(tem.loc[input_index, tem.columns.str.contains('cycle_volume_Down')])
        volume.loc['cycle_volume_Down_{}_{}_31'.format(inner_outer, segment_id)] = volume.loc['cycle_volume_Down_{}_{}_30'.format(inner_outer, segment_id), input_index]
        ts = pd.date_range(start=start_time - timedelta(minutes=31), end=start_time - timedelta(minutes=1), freq='T')[::-1]  # Generate time series
        ts = [t.strftime("%Y-%m-%d %H:%M:%S") for t in ts]
        volume.index = ts
        volume[input_index] = volume[input_index].astype(int)
        volume = volume.rename(columns={input_index: '{}_{}'.format(inner_outer, segment_id)})
        historical_volume = historical_volume.join(volume, on='Time')

        travel_time = pd.DataFrame(tem.loc[input_index, tem.columns.str.contains('avg_travel_time')])
        travel_time.loc['avg_travel_time_{}_{}_31'.format(inner_outer, segment_id)] = travel_time.loc['avg_travel_time_{}_{}_30'.format(inner_outer, segment_id), input_index]

        travel_time.index = ts
        travel_time[input_index] = travel_time[input_index].astype(int)
        travel_time = travel_time.rename(columns={input_index: '{}_{}'.format(inner_outer, segment_id)})
        historical_travel_time = historical_travel_time.join(travel_time, on='Time')

    historical_volume = historical_volume.rename(columns={"0_0": "941", "0_1": "95", "0_2": "3", "0_3": "716", "1_0": "910", "1_1": "818", "1_2": "805", "1_3": "241"})
    historical_volume = historical_volume.set_index('Time')

    historical_travel_time = historical_travel_time.rename(columns={"0_0": "941", "0_1": "95", "0_2": "3", "0_3": "716", "1_0": "910", "1_1": "818", "1_2": "805", "1_3": "241"})
    historical_travel_time = historical_travel_time.set_index('Time')

    return historical_volume, historical_travel_time


def multiplePrediction(input_index, oneWeekInput, oneWeekLable, model, test_time, start, end, folder_path, start_time, data):
    tem = oneWeekLable.reset_index(drop=True)
    tem = tem.drop('Time', 1)
    current_travel_time = tem.iloc[input_index: input_index + 1, :]
    current_travel_time['time'] = test_time.loc[input_index, 'time']
    current_travel_time['predictionHorizon'] = 0

    for predictionHorizon in ([5, 10, 15, 20, 25, 30]):

        tem_model = model['model_{}'.format(predictionHorizon)]
        predictions = getPrediction(input_index, oneWeekInput, tem_model, predictionHorizon, test_time)

        current_travel_time = pd.concat([current_travel_time, predictions])

        print(predictionHorizon)

    current_travel_time = current_travel_time.rename(columns={"y_0_0": "941", "y_0_1": "95", "y_0_2": "3", "y_0_3": "716", "y_1_0": "910", "y_1_1": "818", "y_1_2": "805", "y_1_3": "241"})
    current_travel_time = current_travel_time.reset_index(drop=True)
    current_travel_time['time'] = current_travel_time['time'].astype(str)
    current_travel_time = current_travel_time.set_index('predictionHorizon')
    current_travel_time = current_travel_time.drop(['time'], axis=1)

    # convert travel time to travel time index
    current_travel_time_index = current_travel_time.copy()
    current_travel_time_index['941'] = current_travel_time_index['941']/20.69
    current_travel_time_index['95'] = current_travel_time_index['95']/49.25
    current_travel_time_index['3'] = current_travel_time_index['3']/23.34
    current_travel_time_index['716'] = current_travel_time_index['716']/18.49
    current_travel_time_index['910'] = current_travel_time_index['910']/20.81
    current_travel_time_index['818'] = current_travel_time_index['818']/49.37
    current_travel_time_index['805'] = current_travel_time_index['805']/23.53
    current_travel_time_index['241'] = current_travel_time_index['241']/18.51

    # result = current_travel_time.append(current_travel_time_index, sort=False).reset_index(drop=True)
    #
    # current_travel_time = current_travel_time.join(current_travel_time_index, how='left', lsuffix='_tt', rsuffix='_index')
    #
    # current_travel_time['941'] = current_travel_time["941_tt"].astype(str) + ', ' + current_travel_time["941_index"].astype(str)
    # current_travel_time['95'] = current_travel_time["95_tt"].astype(str) + ', ' + current_travel_time["95_index"].astype(str)
    # current_travel_time['3'] = current_travel_time["3_tt"].astype(str) + ', ' + current_travel_time["3_index"].astype(str)
    # current_travel_time['716'] = current_travel_time["716_tt"].astype(str) + ', ' + current_travel_time["716_index"].astype(str)
    # current_travel_time['910'] = current_travel_time["910_tt"].astype(str) + ', ' + current_travel_time["910_index"].astype(str)
    # current_travel_time['818'] = current_travel_time["818_tt"].astype(str) + ', ' + current_travel_time["818_index"].astype(str)
    # current_travel_time['805'] = current_travel_time["805_tt"].astype(str) + ', ' + current_travel_time["805_index"].astype(str)
    # current_travel_time['241'] = current_travel_time["241_tt"].astype(str) + ', ' + current_travel_time["241_index"].astype(str)
    #
    # result = current_travel_time[['941', '95', '3', '716', '910', '818', '805', '241', 'time_tt']]
    # result = result.rename(columns = {'time_tt': 'time'})

    historical_volume, historical_travel_time = gen_historical_data(start, end, folder_path, input_index, start_time, data)
    historical_volume = historical_volume / 5

    historical_volume = historical_volume.iloc[::-1]
    historical_travel_time = historical_travel_time.iloc[::-1]

    dict1 = current_travel_time.to_dict()
    dict2 = current_travel_time_index.to_dict()
    dict3 = historical_volume.to_dict()
    dict4 = historical_travel_time.to_dict()

    ds = [dict1, dict2, dict3, dict4]
    d = {}
    for k in dict1.keys():
        d[k] = tuple(d[k] for d in ds)

    # dict5 = {}
    # for (k, v, u, t) in list(dict1.items()) + list(dict2.items()) + list(dict3.items()) + list(dict4.items()):
    #     try:
    #         dict5[k] += [v]
    #         dict5[k] += [u]
    #         dict5[k] += [t]
    #     except KeyError:
    #         dict5[k] = [v] + [u] + [t]

    try:
        upload2DB(d, 'http://165.227.124.59:5000/posttraveltimedata')
    except:
        print('Travel Time Prediction:', time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime(time.time())), ' upload data error.')

    return current_travel_time


folder_path = 'Data/split_data/5_min'

data_file = 'Data/modeling_data_v1.h5'
x_train, y_train, x_test, y_test = load_data_for_convLSTM(data_file)

y_label = pd.read_csv('Data/split_data/5_min/y_segment_travel_time.csv', index_col=0).reset_index(drop=True)
y_label['Time'] = pd.to_datetime(y_label['Time'])

start = '2018-11-23 00:00:00'
end = '2018-11-29 23:59:00'
oneWeekLable = y_label.loc[(y_label['Time'] >= datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S') - timedelta(minutes=5)) & (y_label['Time'] <= datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S') - timedelta(minutes=5))]
oneWeekLable.shape

start_index = oneWeekLable.index[0] + 5 - x_train.shape[0]
end_index = oneWeekLable.index[-1] + 5 - x_train.shape[0]
oneWeekInput = x_test[start_index:end_index + 1]
oneWeekInput.shape

ts = pd.date_range(start=start, end=end, freq='T')  # Generate time series
time_reference = ts.to_frame().reset_index(drop=True)
time_reference.rename(columns={0: 'Time'}, inplace=True)

time_reference['day_of_week'] = time_reference['Time'].dt.dayofweek
time_reference['hour'] = time_reference['Time'].dt.hour
time_reference['minute'] = time_reference['Time'].dt.minute
time_reference['time'] = time_reference['Time'].dt.time

test_time = time_reference.loc[:, ['time']]


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    scaler_y = joblib.load('scaler.save')
    model_5 = load_model('Data/Visualization_Platform/my_model_5.h5')
    model_10 = load_model('Data/Visualization_Platform/my_model_10.h5')
    model_15 = load_model('Data/Visualization_Platform/my_model_15.h5')
    model_20 = load_model('Data/Visualization_Platform/my_model_20.h5')
    model_25 = load_model('Data/Visualization_Platform/my_model_25.h5')
    model_30 = load_model('Data/Visualization_Platform/my_model_30.h5')

    model = {'model_5': model_5, 'model_10': model_10, 'model_15': model_15, 'model_20': model_20, 'model_25': model_25, 'model_30': model_30}

    data = read_historical(start, end, folder_path)

    while True:
        start_time = datetime.datetime.now().replace(microsecond=0)
        hours = start_time.hour
        minutes = start_time.minute
        dayOfWeek = start_time.weekday()
        input_index = time_reference.loc[(time_reference['hour'] == hours) & (time_reference['minute'] == minutes) & (time_reference['day_of_week'] == dayOfWeek)].index.tolist()[0]

        predictions = multiplePrediction(input_index, oneWeekInput, oneWeekLable, model, test_time, start, end, folder_path, start_time, data)
        if 60 - (datetime.datetime.now() - start_time).total_seconds() > 0:
            print('Travel Time Prediction:', datetime.datetime.now(), ' Whole processing costs:', int((datetime.datetime.now() - start_time).total_seconds()), 'seconds.')
            time.sleep(60 - int((datetime.datetime.now() - start_time).total_seconds()))
        else:
            print('Travel Time Prediction:', datetime.datetime.now(), ' Whole processing costs:', int((datetime.datetime.now() - start_time).total_seconds()), 'seconds.')
            continue


