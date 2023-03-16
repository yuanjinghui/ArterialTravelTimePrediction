"""
Created on Mon Jul 21 20:01:56 2019

@author: ji758507
"""
# ==========================================================================================================
# ARIMA
# ==========================================================================================================
import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
from multiprocessing import Pool


# ARIMA tuning
def evaluation_metrics(test, predictions):
    mse = mean_squared_error(test, predictions)

    mape_sum = sum((abs((test - predictions)) / test))
    mape = mape_sum / len(test)

    mae_sum = sum(abs(test - predictions))
    mae = mae_sum / len(test)

    return mse, mape, mae


def evaluate_arima_model(X, arima_order, prediction_horizon):
    # prepare training dataset
    train_size = int(len(X) * 0.9999)
    train, test = X[0:train_size-prediction_horizon], X[train_size:]
    history = [x for x in train]
    print(len(test))
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast(steps=prediction_horizon)[0][prediction_horizon-1]
        predictions.append(yhat)
        history.append(X[train_size - prediction_horizon + t])
        print(t)
    # calculate out of sample error
    mse, mape, mae = evaluation_metrics(test, predictions)

    return mse, mape, mae, test, predictions


def evaluate_models(dataset, p_values, d_values, q_values, prediction_horizon):
    dataset = dataset.astype('float32')
    best_mape, best_mse, best_mae, best_p, best_d, best_q = float("inf"), float("inf"), float("inf"), None, None, None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse, mape, mae = evaluate_arima_model(dataset, order, prediction_horizon)
                    if mape < best_mape:
                        best_mape, best_mse, best_mae, best_p, best_d, best_q = mape, mse, mae, p, d, q
                    print('ARIMA%s MSE=%.3f' % (order, mse, mape, mae))
                except:
                    continue
    print('Best ARIMA%s %s %s MAPE=%.4f MSE=%.4f MAE=%.4f' % (best_p, best_d, best_q, best_mape, best_mse, best_mae))
    return best_p, best_d, best_q, best_mape, best_mse, best_mae


def tune_arima(y_label,  p_values, d_values, q_values, prediction_horizon):
    # y_label = input_data[0]
    # p_values = input_data[1]
    # d_values = input_data[2]
    # q_values = input_data[3]
    # prediction_horizon = input_data[4]
    total_test = list()
    total_predict = list()
    performance = []
    for inner_outer in range(2):
        for i in range(4):
            y = y_label.loc[:, 'y_{}_{}'.format(inner_outer, i)]
            # best_p, best_d, best_q, best_mape, best_mse, best_mae = evaluate_models(y.values, p_values, d_values, q_values, prediction_horizon)
            # performance.append({'inner_outer': inner_outer, 'i': i, 'best_p': best_p, 'best_d': best_d, 'best_q': best_q, 'mae': best_mae, 'mape': best_mape, 'mse': best_mse})
            order = (p_values, d_values, q_values)
            mse, mape, mae, test, predictions = evaluate_arima_model(y.values, order, prediction_horizon)
            print(inner_outer, i, mse, mape, mae)
            performance.append({'inner_outer': inner_outer, 'i': i, 'mae': mae, 'mape': mape, 'mse': mse})

            total_predict.append(predictions)
            total_test.append(test)

    print(total_predict.shape, total_test.shape)
    total_test = np.asarray(total_test).ravel()
    total_predict = np.asarray(total_predict).ravel()
    print(total_predict.shape, total_test.shape)

    overall_mse, overall_mape, overall_mae = evaluation_metrics(total_test, total_predict)
    performance.append({'inner_outer': 3, 'i': 4, 'mae': overall_mae, 'mape': overall_mape, 'mse': overall_mse})
    print(overall_mse, overall_mape, overall_mae)
    performance = pd.DataFrame(performance)
    return performance


# class Bar(object):
#     def __init__(self, x):
#         self.x = x