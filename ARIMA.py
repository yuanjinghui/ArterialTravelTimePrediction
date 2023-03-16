"""
Created on Mon Jul 21 20:01:56 2019

@author: Jinghui Yuan: jinghuiyuan@knights.ucf.edu
"""
# ==========================================================================================================
# ARIMA
# ==========================================================================================================
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import warnings
import math
import matplotlib.pyplot as plt


# ARIMA tuning
def evaluation_metrics(test, predictions):
    mse_sum = sum((test - predictions) ** 2)
    rmse = math.sqrt(mse_sum / len(test))

    mape_sum = sum(abs(test - predictions) / test)
    mape = mape_sum / len(test)

    mae_sum = sum(abs(test - predictions))
    mae = mae_sum / len(test)

    return rmse, mape, mae


def evaluate_arima_model(X, arima_order, prediction_horizon, train_test_split):
    # prepare training dataset
    train_size = int(len(X) * train_test_split)
    train, test = X[0:train_size-prediction_horizon], X[train_size:]

    # to save the processing time, only 1440 samples from train dataset were utilized
    history = [x for x in train[-1440:]]
    print(len(test))
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history[t:], order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast(steps=prediction_horizon)[0][prediction_horizon-1]
        predictions.append(yhat)
        history.append(X[train_size - prediction_horizon + t])
        print(t)
    # calculate out of sample error
    rmse, mape, mae = evaluation_metrics(test, predictions)

    return rmse, mape, mae, test, predictions


def evaluate_models(dataset, p_values, d_values, q_values, prediction_horizon):
    dataset = dataset.astype('float32')
    best_mape, best_mse, best_mae, best_p, best_d, best_q = float("inf"), float("inf"), float("inf"), None, None, None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse, mape, mae = evaluate_arima_model(dataset, order, prediction_horizon)
                    if mape < best_mape:
                        best_mape, best_mse, best_mae, best_p, best_d, best_q = mape, rmse, mae, p, d, q
                    print('ARIMA%s MSE=%.3f' % (order, rmse, mape, mae))
                except:
                    continue
    print('Best ARIMA%s %s %s MAPE=%.4f MSE=%.4f MAE=%.4f' % (best_p, best_d, best_q, best_mape, best_mse, best_mae))
    return best_p, best_d, best_q, best_mape, best_mse, best_mae


def tune_arima(y_label,  p_values, d_values, q_values, prediction_horizon, train_test_split):
    # y_label = input_data[0]
    # p_values = input_data[1]
    # d_values = input_data[2]
    # q_values = input_data[3]
    # prediction_horizon = input_data[4]

    num_sample_split = int(len(y_label) * train_test_split)
    # split train and test dataset by 80:20
    y_test = y_label.iloc[num_sample_split:, :].reset_index(drop=True)
    test_time = y_test.loc[:, ['Time']]

    total_test = list()
    total_predict = list()
    performance = []
    for inner_outer in range(2):
        for i in range(4):
            y = y_label.loc[:, 'y_{}_{}'.format(inner_outer, i)]
            # best_p, best_d, best_q, best_mape, best_mse, best_mae = evaluate_models(y.values, p_values, d_values, q_values, prediction_horizon)
            # performance.append({'inner_outer': inner_outer, 'i': i, 'best_p': best_p, 'best_d': best_d, 'best_q': best_q, 'mae': best_mae, 'mape': best_mape, 'mse': best_mse})
            order = (p_values, d_values, q_values)
            rmse, mape, mae, test, predictions = evaluate_arima_model(y.values, order, prediction_horizon, train_test_split)
            print(inner_outer, i, rmse, mape, mae)
            performance.append({'inner_outer': inner_outer, 'i': i, 'mae': mae, 'mape': mape, 'rmse': rmse})

            total_predict.append(predictions)
            total_test.append(test)

            test_prediction = pd.DataFrame({'Time': test_time.Time, 'y_{}_{}_test'.format(inner_outer, i): test, 'y_{}_{}_pred'.format(inner_outer, i): predictions})
            test_time = test_time.join(test_prediction.set_index('Time'), on='Time')

    # print(total_predict.shape, total_test.shape)
    total_test = np.asarray(total_test).ravel()
    total_predict = np.asarray(total_predict).ravel()
    print(total_predict.shape, total_test.shape)

    overall_rmse, overall_mape, overall_mae = evaluation_metrics(total_test, total_predict)
    performance.append({'inner_outer': 3, 'i': 4, 'mae': overall_mae, 'mape': overall_mape, 'rmse': overall_rmse})

    print(overall_rmse, overall_mape, overall_mae)
    performance = pd.DataFrame(performance)
    return performance, test_time


def prediction_plot(best_model_predictions, root_path):

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

    fig.savefig(root_path + '/Comparison.png', transparent=False, dpi=400)
# inner_outer = 0
# i = 1
# p = 1
# d = 0
# q = 1
# X = y.values
# arima_order = order
# evaluate parameters
p_values = 5
d_values = 1
q_values = 0
# evaluate_models(series.values, p_values, d_values, q_values)
prediction_horizon = 30
y_label = pd.read_csv('Data/split_data/30_min/y_segment_travel_time.csv', index_col=0)
warnings.filterwarnings("ignore")
train_test_split = 0.8
if __name__ == '__main__':
    # pool = Pool(processes=8)
    # input_data = (y_label, p_values, d_values, q_values, prediction_horizon)
    # input[4]
    # performance = pool.starmap(ARIMA_functions.tune_arima, [ARIMA_functions.Bar(y_label),  ARIMA_functions.Bar(p_values), ARIMA_functions.Bar(d_values), ARIMA_functions.Bar(q_values), ARIMA_functions.Bar(prediction_horizon)])
    # performance = pool.map(ARIMA_functions.tune_arima, input_data)
    performance, prediction = tune_arima(y_label,  p_values, d_values, q_values, prediction_horizon, train_test_split)
    performance.to_csv('Data/Save_models/ARIMA/30_min/arima_performance.csv', sep=',')
    prediction.to_csv('Data/Save_models/ARIMA/30_min/arima_prediction.csv', sep=',')

    plot = prediction[(prediction['Time'] >= '2018-11-19 00:00:00') & (prediction['Time'] <= '2018-11-20 00:00:00')].reset_index(drop=True)
    prediction_plot(plot, 'Data/Save_models/ARIMA/30_min')


