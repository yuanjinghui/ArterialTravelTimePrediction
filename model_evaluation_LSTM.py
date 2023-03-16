"""
Created on Mon Jul 21 20:01:56 2019

@author: ji758507
"""
import h5py
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import math

# ini_array1 = np.array([[1, 2, 3], [2, 4, 5]])
# ini_array1.shape
# a = ini_array1.ravel()
# a.shape
# b = a.reshape(2, 3)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use id from $ nvidia-smi


def label_reshape(y_pred):
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    scaler_filename = "scaler.save"
    scaler = joblib.load(scaler_filename)
    y_pred_r = scaler.inverse_transform(y_pred)
    # 2d array to 1d array
    y_pred_r = y_pred_r.ravel()
    return y_pred_r


def label_reshape_cnn_lstm(y_pred, num_samples):
    y_pred = y_pred.reshape(num_samples, -1)
    scaler_filename = "scaler.save"
    scaler = joblib.load(scaler_filename)
    y_pred_r = scaler.inverse_transform(y_pred)
    # 2d array to 1d array
    y_pred_r = y_pred_r.ravel()
    return y_pred_r


def x_reshape(x_train):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], -1)
    a = x_train.shape[1]
    b = x_train.shape[2]
    x_train = np.moveaxis(x_train, -1, 1)
    x_train = x_train.reshape(-1, a, b)
    return x_train


def evaluation_metrics(test, predictions):
    mse_sum = sum((test - predictions) ** 2)
    rmse = math.sqrt(mse_sum / len(test))

    mape_sum = sum(abs(test - predictions) / test)
    mape = mape_sum / len(test)

    mae_sum = sum(abs(test - predictions))
    mae = mae_sum / len(test)

    return rmse, mape, mae


def load_data_for_convLSTM(data_file):
    # with h5py.File('Data/modeling_data_15_v1.h5', 'r') as f:
    # with h5py.File('Data/modeling_data_10_v1.h5', 'r') as f:
    # with h5py.File('Data/modeling_data_v1.h5', 'r') as f:
    with h5py.File(data_file, 'r') as f:
        print(list(f.keys()))
        x_test = f['x_test'][:]
        y_test = f['y_test'][:]
        f.close()
    return x_test, y_test


def gen_prediction(y_test_r, y_pred_r, num_samples, test_time):
    y_test_r = y_test_r.reshape(num_samples, 4, 2)
    y_pred_r = y_pred_r.reshape(num_samples, 4, 2)
    for inner_outer in range(2):
        for i in range(4):
            prediction = pd.DataFrame({'Time': test_time.Time, 'y_{}_{}_test'.format(inner_outer, i): y_test_r[:, i, inner_outer], 'y_{}_{}_pred'.format(inner_outer, i): y_pred_r[:, i, inner_outer]})
            test_time = test_time.join(prediction.set_index('Time'), on='Time')

            print(inner_outer, i)

    return test_time


def model_comparison(root_path, test_time, model_type, data_file, model_folder):

    performance = []
    x_test, y_test = load_data_for_convLSTM(data_file)
    print(x_test.shape, y_test.shape)
    print(model_type)

    # for model_type in (['ConvLSTM', 'CNN_LSTM', 'LSTM']):

    # get the folder path for models
    model_path = os.path.join(root_path + model_type + '/' + model_folder)
    # model_path = os.path.join(root_path)
    files = os.listdir(model_path)
    models = [i for i in files if 'h5' in i]
    print(models)

    # for ConvLSTM, the prediction need to be reshaped
    if model_type == 'ConvLSTM':
        print('ConvLSTM')

        for i in range(len(models)):
            model_name = models[i]
            print(model_name)
            model = load_model(os.path.join(model_path + '/' + model_name))
            y_pred = model.predict(x_test)

            num_samples = y_pred.shape[0]
            del model
            y_pred_r = label_reshape(y_pred)
            y_test_r = label_reshape(y_test)
            print(y_test_r.shape, y_pred_r.shape)

            rmse, mape, mae = evaluation_metrics(y_test_r, y_pred_r)

            prediction = gen_prediction(y_test_r, y_pred_r, num_samples, test_time)

            prediction.to_csv(os.path.join(model_path + '/' + 'Prediction_performance_{}_{}.csv'.format(model_type, i)), sep=',')
            performance.append({'model_type': model_type, 'model': model_name, 'sub_model_id': i, 'mae': mae, 'mape': mape, 'rmse': rmse})
            print(model_type, i, model_name, mae, mape, rmse)

    elif model_type == 'CNN_LSTM':
        print('CNN_LSTM')

        num_samples = x_test.shape[0]
        x_test_r = x_reshape(x_test)
        y_test_r = label_reshape(y_test)

        n_features = x_test_r.shape[2]
        n_steps, n_length = 5, 6
        x_test_r = x_test_r.reshape((x_test_r.shape[0], n_steps, n_length, n_features))

        for i in range(len(models)):
            model_name = models[i]
            model = load_model(os.path.join(model_path + '/' + model_name))
            y_pred = model.predict(x_test_r)

            del model
            y_pred_r = label_reshape_cnn_lstm(y_pred, num_samples)
            print(y_test_r.shape, y_pred_r.shape)

            rmse, mape, mae = evaluation_metrics(y_test_r, y_pred_r)

            prediction = gen_prediction(y_test_r, y_pred_r, num_samples, test_time)

            prediction.to_csv(os.path.join(model_path + '/' + 'Prediction_performance_{}_{}.csv'.format(model_type, i)), sep=',')
            performance.append({'model_type': model_type, 'model': model_name, 'sub_model_id': i, 'mae': mae, 'mape': mape, 'rmse': rmse})
            print(model_type, i, model_name, mae, mape, rmse)

    elif model_type == 'LSTM':
        print('LSTM')
        num_samples = x_test.shape[0]
        x_test_r = x_reshape(x_test)
        y_test_r = label_reshape(y_test)

        for i in range(len(models)):
            model_name = models[i]
            model = load_model(os.path.join(model_path + '/' + model_name))
            y_pred = model.predict(x_test_r)
            print(y_test_r.shape, y_pred.shape)

            del model
            y_pred_r = label_reshape_cnn_lstm(y_pred, num_samples)
            rmse, mape, mae = evaluation_metrics(y_test_r, y_pred_r)

            prediction = gen_prediction(y_test_r, y_pred_r, num_samples, test_time)

            prediction.to_csv(os.path.join(model_path + '/' + 'Prediction_performance_{}_{}.csv'.format(model_type, i)), sep=',')
            performance.append({'model_type': model_type, 'model': model_name, 'sub_model_id': i, 'mae': mae, 'mape': mape, 'rmse': rmse})
            print(model_type, i, model_name, mae, mape, rmse)

    elif model_type == 'GRU':
        print('GRU')
        num_samples = x_test.shape[0]
        x_test_r = x_reshape(x_test)
        y_test_r = label_reshape(y_test)

        for i in range(len(models)):
            model_name = models[i]
            model = load_model(os.path.join(model_path + '/' + model_name))
            y_pred = model.predict(x_test_r)
            print(y_test_r.shape, y_pred.shape)

            del model
            y_pred_r = label_reshape_cnn_lstm(y_pred, num_samples)
            rmse, mape, mae = evaluation_metrics(y_test_r, y_pred_r)

            prediction = gen_prediction(y_test_r, y_pred_r, num_samples, test_time)

            prediction.to_csv(os.path.join(model_path + '/' + 'Prediction_performance_{}_{}.csv'.format(model_type, i)), sep=',')
            performance.append({'model_type': model_type, 'model': model_name, 'sub_model_id': i, 'mae': mae, 'mape': mape, 'rmse': rmse})
            print(model_type, i, model_name, mae, mape, rmse)


    performance = pd.DataFrame(performance)
        # performance.to_csv(os.path.join(root_path + '/' + 'overall_performance_v3.csv'), sep=',')

    return performance


def evaluation(prediction):

    total_test = list()
    total_predict = list()
    performance = []
    for inner_outer in range(2):
        for i in range(4):
            test = prediction.loc[:, 'y_{}_{}_test'.format(inner_outer, i)]
            predict = prediction.loc[:, 'y_{}_{}_pred'.format(inner_outer, i)]
            rmse, mape, mae = evaluation_metrics(test, predict)
            print(inner_outer, i, rmse, mape, mae)
            performance.append({'inner_outer': inner_outer, 'i': i, 'mae': mae, 'mape': mape, 'rmse': rmse})

            total_predict.append(predict)
            total_test.append(test)

    total_test = np.asarray(total_test).ravel()
    total_predict = np.asarray(total_predict).ravel()
    print(total_predict.shape, total_test.shape)

    overall_rmse, overall_mape, overall_mae = evaluation_metrics(total_test, total_predict)
    performance.append({'inner_outer': 3, 'i': 4, 'mae': overall_mae, 'mape': overall_mape, 'rmse': overall_rmse})

    print(overall_rmse, overall_mape, overall_mae)
    performance = pd.DataFrame(performance)

    return performance


def prediction_plot(best_model_predictions, root_path, model_type, model_folder):

    fig = plt.figure(figsize=(26, 20))
    plt.rcParams.update({'font.size': 14})

    for i in range(4):
        for inner_outer in range(2):

            index = 2 * i + 1 + inner_outer
            ax = fig.add_subplot(4, 2, index)
            test = pd.concat([best_model_predictions.Time, best_model_predictions.iloc[:, best_model_predictions.columns.str.contains('_{}_{}'.format(inner_outer, i))]], axis=1)

            # test = test.set_index('Time')
            ax.plot_date(test['Time'], test['y_{}_{}_test'.format(inner_outer, i)], color='black', fmt='b-', linewidth=2.0, label='True')
            ax.plot_date(test['Time'], test['y_{}_{}_pred'.format(inner_outer, i)], color='green', fmt='b-', linewidth=2.0, label='Predicted')
            ax.legend(loc="lower right")

            # plt.savefig('./Data/test.png', dpi=300)
            # plt.show()
            # plt.close()
    fig.savefig(root_path + model_type + '/' + model_folder + '/' + '/Comparison.png', transparent=False, dpi=400)


root_path = 'Data/Save_models/'
# model_type = 'ConvLSTM'
# data_file = 'Data/modeling_data_15.h5'
# model_folder = 'All_variables_server_15_min'
# performance = pd.read_csv(os.path.join(root_path + '\\' + 'overall_performance.csv'), index_col=0)


if __name__ == '__main__':

    # for prediction_horizon in ([20, 25, 30]):
    for prediction_horizon in ([30]):

        y_label = pd.read_csv('Data/split_data/{}_min/y_segment_travel_time.csv'.format(prediction_horizon), index_col=0)
        num_sample_split = int(len(y_label) * 0.8)
        # split train and test dataset by 80:20
        y_test = y_label.iloc[num_sample_split:, :].reset_index(drop=True)
        test_time = y_test.loc[:, ['Time']]

        # for variables in (['All_variables', 'Two_variables']):
        for variables in (['Two_variables']):
            if variables == 'Two_variables':
                data_file = 'Data/modeling_data_{}_v1.h5'.format(prediction_horizon)

            if variables == 'All_variables':
                data_file = 'Data/modeling_data_{}.h5'.format(prediction_horizon)

            model_folder = '{}_server_{}_min'.format(variables, prediction_horizon)

            # for model_type in (['ConvLSTM', 'CNN_LSTM', 'LSTM', 'GRU']):
            for model_type in (['GRU']):
                # call the function
                comparison_performance = model_comparison(root_path, test_time, model_type, data_file, model_folder)

                # concatenate the prediction performance between different sections of the corridor
                comparison_performance["rank"] = comparison_performance.groupby(by=['model_type'], as_index=False)["mape"].rank(ascending=True)
                comparison_performance = comparison_performance.sort_values(by=['rank']).reset_index(drop=True)
                comparison_performance.to_csv(root_path + model_type + '/' + model_folder + '/' + '{}_tuning_performance.csv'.format(model_type), sep=',')

                best_model_id = comparison_performance.loc[comparison_performance['rank'] == 1]['sub_model_id'][0]

                best_model_predictions = pd.read_csv(os.path.join(root_path + model_type + '/' + model_folder + '/' + 'Prediction_performance_{}_{}.csv'.format(model_type, best_model_id)), index_col=0)
                best_model_predictions['Time'] = pd.to_datetime(best_model_predictions['Time'])

                performance = evaluation(best_model_predictions)
                performance.to_csv(root_path + model_type + '/' + model_folder + '/' + '{}_performance.csv'.format(model_type), sep=',')

                plot = best_model_predictions[(best_model_predictions['Time'] >= '2018-11-19 00:00:00') & (best_model_predictions['Time'] <= '2018-11-20 00:00:00')].reset_index(drop=True)
                prediction_plot(plot, root_path, model_type, model_folder)

                print(model_type, variables, '{}_min'.format(prediction_horizon))

