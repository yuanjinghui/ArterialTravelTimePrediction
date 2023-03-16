"""
Created on Fri Jan 17 20:01:56 2020

@author: ji758507
"""
import pandas as pd
from numpy import stack
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


def load_file(file_path):

    temp = pd.read_csv(file_path, index_col=0)
    temp['Time'] = pd.to_datetime(temp['Time'])
    temp = temp.reset_index(drop=True)

    # trick:  avoid misleading recognition of 30 to 3, 20 to 2, 10 to 1
    temp = temp.add_suffix('_')
    temp.rename(columns={'Time_': 'Time'}, inplace=True)

    return temp


def y_to_array(test):

    cols = list()
    for inner_outer in range(2):
        trainy_temp = test.filter(regex='y_{}'.format(inner_outer))
        rows = list()
        for i in range(trainy_temp.shape[1]):
            temp = trainy_temp.loc[:, 'y_{}_{}'.format(inner_outer, i)]
            print(temp.shape)
            rows.append(temp)

        rows = stack(rows, axis=1)
        print(rows.shape)
        cols.append(rows)
    cols = stack(cols, axis=2)
    print(cols.shape)

    return cols


def split_train_test_label(y_label, num_sample_split):

    # split train and test dataset by 80:20
    y_label_train = y_label[:num_sample_split].reset_index(drop=True)
    y_label_test = y_label[num_sample_split:].reset_index(drop=True)

    y_label_train = y_to_array(y_label_train)
    y_label_test = y_to_array(y_label_test)

    # min-max normalization
    scaler_y = MinMaxScaler()
    y_label_train = scaler_y.fit_transform(y_label_train.reshape(y_label_train.shape[0], -1))
    y_label_train = y_label_train.reshape(y_label_train.shape[0], 1, 4, 2)

    y_label_test = scaler_y.transform(y_label_test.reshape(y_label_test.shape[0], -1))
    y_label_test = y_label_test.reshape(y_label_test.shape[0], 1, 4, 2)

    scaler_filename = "scaler.save"
    joblib.dump(scaler_y, scaler_filename)

    return y_label_train, y_label_test


def gen_dataset(SR436_Selected_Segments, time_steps, y_label):

    # for every group, save the numpy train and test data to hdf5
    # all variables 5 min
    # with h5py.File('Data/modeling_data.h5', 'a') as hf:
    # two variables 5 min
    # with h5py.File('Data/modeling_data_v1.h5', 'a') as hf:
    # # all variables 10 min
    # with h5py.File('Data/modeling_data_10.h5', 'a') as hf:
    # two variables 10 min
    # with h5py.File('Data/modeling_data_10_v1.h5', 'a') as hf:
    # all variables 15 min
    # with h5py.File('Data/modeling_data_15.h5', 'a') as hf:
    # two variables 15 min
    # with h5py.File('Data/modeling_data_15_v1.h5', 'a') as hf:
    # two variables 20 min
    # with h5py.File('Data/modeling_data_20_v1.h5', 'a') as hf:
    # two variables 20 min
    # with h5py.File('Data/modeling_data_20.h5', 'a') as hf:
    # two variables 25 min
    # with h5py.File('Data/modeling_data_25_v1.h5', 'a') as hf:
    # with h5py.File('Data/modeling_dat?a_25.h5', 'a') as hf:
    # two variables 30 min
    # with h5py.File('Data/modeling_data_30_v1.h5', 'a') as hf:
    with h5py.File('Data/modeling_data_30.h5', 'a') as hf:
        train_cols = list()
        test_cols = list()
        for inner_outer in range(2):

            segments = SR436_Selected_Segments[SR436_Selected_Segments['inner_outer'] == inner_outer].sort_values(by=['ES_seq']).reset_index(drop=True)

            train_rows = list()
            test_rows = list()
            for i in range(len(segments)):

                pair_id = segments.loc[i, 'Pair_ID']
                # read csv using dask.dataframe and then convert it to pandas dataframe
                temp = load_file('Data/split_data/30_min/segment_data_{}_{}_{}.csv'.format(inner_outer, i, pair_id))

                # variable selection
                # temp = temp.loc[:, temp.columns.str.contains('|'.join(['Time', 'avg_speed', 'avg_travel_time']))]
                # temp.dtypes

                num_sample_split = int(len(temp) * 0.8)
                split_date = temp.loc[num_sample_split, 'Time']
                # split train and test dataset by 80:20
                temp_train = temp.iloc[:num_sample_split, :].reset_index(drop=True)
                temp_test = temp.iloc[num_sample_split:, :].reset_index(drop=True)

                train_time_slices = list()
                test_time_slices = list()

                for key in range(time_steps):
                    # stack from old to new
                    time = time_steps - key
                    # train data
                    train = temp_train.loc[:, temp_train.columns.str.contains('_{}_{}_{}_'.format(inner_outer, i, time))]

                    # min-max normalization
                    scaler_x = MinMaxScaler()
                    train = scaler_x.fit_transform(train)

                    print(train.shape)
                    train_time_slices.append(train)

                    # test data
                    test = temp_test.loc[:, temp_test.columns.str.contains('_{}_{}_{}_'.format(inner_outer, i, time))]
                    test = scaler_x.transform(test)

                    print(test.shape)
                    test_time_slices.append(test)

                # stack over time slices to generate array with the shape of (samples, time_steps, num_features)
                train_time_slices = stack(train_time_slices, axis=1)
                print(train_time_slices.shape)
                test_time_slices = stack(test_time_slices, axis=1)
                print(test_time_slices.shape)

                # append over number of rows
                train_rows.append(train_time_slices)
                test_rows.append(test_time_slices)

                # stack over number of rows to generate array with the shape of (samples, time_steps, num_features, rows)
            train_rows = stack(train_rows, axis=3)
            print(train_rows.shape)
            test_rows = stack(test_rows, axis=3)
            print(test_rows.shape)

            # append over number of cols
            train_cols.append(train_rows)
            test_cols.append(test_rows)

        # stack over number of cols to generate array with the shape of (samples, time_steps, num_features, rows, cols)
        train_cols = stack(train_cols, axis=4)
        print(train_cols.shape)
        test_cols = stack(test_cols, axis=4)
        print(test_cols.shape)

        num_features = train_cols.shape[2]
        num_sample_split = train_cols.shape[0]

        # process the y label data for specific group
        y_label_train, y_label_test = split_train_test_label(y_label, num_sample_split)

        # for every group, save the numpy train and test data to hdf5
        hf.create_dataset('x_train', data=train_cols)
        print('x_train', train_cols.shape)

        hf.create_dataset('y_train', data=y_label_train)
        print('y_train', y_label_train.shape)

        # test dataset
        hf.create_dataset('x_test', data=test_cols)
        print('x_test', test_cols.shape)

        hf.create_dataset('y_test', data=y_label_test)
        print('y_test', y_label_test.shape)

        del train_cols, test_cols, y_label_train, y_label_test


SR436_Selected_Segments = pd.read_csv('Data/SR436_Selected_Segments.csv')
intersection_approaches_dict = pd.read_csv('J:\ATSPM_CNN_LSTM\Data\intersection_approaches_dict.csv')
SR436_Selected_Segments = pd.merge(SR436_Selected_Segments, intersection_approaches_dict[['Seg_ID', 'ES_seq', 'inner_outer']], how='left', left_on=['Seg_ID'], right_on=['Seg_ID'])

y_label = pd.read_csv('Data/split_data/30_min/y_segment_travel_time.csv', index_col=0)


# inner_outer = 0
# i = 0
# time_steps = 30
# key = 27
# temp.dtypes
if __name__ == '__main__':
    gen_dataset(SR436_Selected_Segments, 30, y_label)
