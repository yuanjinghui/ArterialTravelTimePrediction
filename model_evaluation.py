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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use id from $ nvidia-smi


def label_reshape(y_pred):
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    scaler_filename = "scaler.save"
    scaler = joblib.load(scaler_filename)
    y_pred_r = scaler.inverse_transform(y_pred)
    y_pred_r = y_pred_r.ravel()
    return y_pred_r


def evaluation_metrics(test, predictions):
    mse_sum = sum((test - predictions) ** 2)
    rmse = math.sqrt(mse_sum / len(test))

    mape_sum = sum(abs(test - predictions) / test)
    mape = mape_sum / len(test)

    mae_sum = sum(abs(test - predictions))
    mae = mae_sum / len(test)

    return rmse, mape, mae


def load_data_for_convLSTM():
    # with h5py.File('Data/modeling_data_v1.h5', 'r') as f:
    with h5py.File('Data/modeling_data.h5', 'r') as f:
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


def model_comparison(root_path, test_time):

    performance = []
    x_test, y_test = load_data_for_convLSTM()
    print(x_test.shape, y_test.shape)
    model_type = 'ConvLSTM'
    # for model_type in (['ConvLSTM', 'CNN_LSTM', 'LSTM']):

    # get the folder path for models
    # model_path = os.path.join(root_path + '\\' + model_type)
    model_path = os.path.join(root_path)
    files = os.listdir(model_path)
    models = [i for i in files if 'h5' in i]
    print(models)

    # for ConvLSTM, the prediction need to be reshaped
    # if model_type == 'ConvLSTM':

    for i in range(len(models)):
        model_name = models[i]
        print(model_name)
        model = load_model(os.path.join(model_path + '/' + model_name))
        y_pred = model.predict(x_test)

        num_samples = y_pred.shape[0]
        del model
        y_pred_r = label_reshape(y_pred)
        y_test_r = label_reshape(y_test)
        rmse, mape, mae = evaluation_metrics(y_test_r, y_pred_r)

        prediction = gen_prediction(y_test_r, y_pred_r, num_samples, test_time)

        prediction.to_csv(os.path.join(root_path + '/' + 'Prediction_performance_{}_{}.csv'.format(model_type, i)), sep=',')
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


def prediction_plot(best_model_predictions, root_path):

    fig = plt.figure(figsize=(26, 20))
    plt.rcParams.update({'font.size': 14})

    for inner_outer in range(2):
        for i in range(4):

            index = 4 * inner_outer + 1 + i
            ax = fig.add_subplot(4, 2, index)
            test = pd.concat([best_model_predictions.Time, best_model_predictions.iloc[:, best_model_predictions.columns.str.contains('_{}_{}'.format(inner_outer, i))]], axis=1)

            # test = test.set_index('Time')
            ax.plot_date(test['Time'], test['y_{}_{}_test'.format(inner_outer, i)], color='black', fmt='b-', linewidth=2.0, label='True')
            ax.plot_date(test['Time'], test['y_{}_{}_pred'.format(inner_outer, i)], color='green', fmt='b-', linewidth=2.0, label='Predicted')
            ax.legend(loc="lower right")

            # plt.savefig('./Data/test.png', dpi=300)
            # plt.show()
            # plt.close()
    fig.savefig(root_path + '/Comparison.png', transparent=False, dpi=400)


# model_type = 'ConvLSTM'
# i = 0
# y_pred.shape
# y_test_r.shape
# plot
root_path = 'Data/Save_models/ConvLSTM'

# performance = pd.read_csv(os.path.join(root_path + '\\' + 'overall_performance.csv'), index_col=0)

y_label = pd.read_csv('Data/split_data/y_segment_travel_time.csv', index_col=0)
num_sample_split = int(len(y_label) * 0.8)
# split train and test dataset by 80:20
y_test = y_label.iloc[num_sample_split:, :].reset_index(drop=True)
test_time = y_test.loc[:, ['Time']]


if __name__ == '__main__':
    # call the function
    comparison_performance = model_comparison(root_path, test_time)

    # concatenate the prediction performance between different sections of the corridor
    comparison_performance["rank"] = comparison_performance.groupby(by=['model_type'], as_index=False)["mape"].rank(ascending=True)
    comparison_performance = comparison_performance.sort_values(by=['rank']).reset_index(drop=True)

    best_model_id = comparison_performance.loc[comparison_performance['rank'] == 1]['sub_model_id'][0]

    best_model_predictions = pd.read_csv(os.path.join(root_path + '/' + 'Prediction_performance_ConvLSTM_{}.csv'.format(best_model_id)), index_col=0)
    best_model_predictions['Time'] = pd.to_datetime(best_model_predictions['Time'])

    performance = evaluation(best_model_predictions)
    performance.to_csv('Data/Save_models/ConvLSTM/ConvLSTM_performance.csv', sep=',')

    plot = best_model_predictions[(best_model_predictions['Time'] >= '2018-11-19 00:00:00') & (best_model_predictions['Time'] <= '2018-11-20 00:00:00')].reset_index(drop=True)
    prediction_plot(plot, root_path)
# test.dtypes
# inner_outer = 0
# i = 0
# test.describe()
# group = 0
# seg_type = 'intersection'
# model_type = 'LSTM'
# i = 0
# model = 'my_model_2500_20_0.4400000000000001_SGD.h5'
#
# model = load_model('./Data/Save_models/Group2_intersection/my_model_2500_20_0.4400000000000001_Adadelta.h5')
# y_pred = model.predict(x_test)
#
# y_pred_r = label_reshape(y_pred)
# y_test_r = label_reshape(y_test)
#
# auc, sensitivity, false_AR, conf_matrix, prediction = evaluation(y_test_r, y_pred_r)


