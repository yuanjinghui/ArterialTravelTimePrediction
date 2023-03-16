"""
Created on Mon Jul 21 20:01:56 2019

@author: Jinghui Yuan: jinghuiyuan@knights.ucf.edu
"""
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import optunity
import optunity.metrics
import sklearn
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib


def load_data_for_convLSTM():
    with h5py.File('Data/modeling_data_10_v1.h5', 'r') as f:
    # with h5py.File('Data/modeling_data_v1.h5', 'r') as f:
        print(list(f.keys()))
        x_train = f['x_train'][:]
        y_train = f['y_train'][:]
        x_test = f['x_test'][:]
        y_test = f['y_test'][:]
        f.close()
    return x_train, y_train, x_test, y_test


def label_reshape(y_pred):
    y_pred = y_pred.reshape(y_pred.shape[0], 1, -1)
    y_pred_r = np.moveaxis(y_pred, -1, 1)
    y_pred_r = y_pred_r.reshape(-1, 1)
    return y_pred_r


def x_reshape(x_train):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], -1)
    x_train_r = np.moveaxis(x_train, -1, 1)
    x_train_r = x_train_r.reshape(-1, x_train.shape[1], x_train.shape[2])
    x_train_r = x_train_r.reshape(x_train_r.shape[0], -1)
    return x_train_r


# convert normalized label to real value
def label_reverse(y_pred, num_samples):
    y_pred = y_pred.reshape(num_samples, -1)
    scaler_filename = "scaler.save"
    scaler = joblib.load(scaler_filename)
    y_pred_r = scaler.inverse_transform(y_pred)
    # 2d array to 1d array
    y_pred_r = y_pred_r.ravel()
    return y_pred_r


def load_file(file_path):

    temp = pd.read_csv(file_path, index_col=0)
    temp['Time'] = pd.to_datetime(temp['Time'])
    temp = temp.reset_index(drop=True)

    return temp


def evaluation_metrics(test, predictions):
    mse_sum = sum((test - predictions) ** 2)
    rmse = math.sqrt(mse_sum / len(test))

    mape_sum = sum(abs(test - predictions) / test)
    mape = mape_sum / len(test)

    mae_sum = sum(abs(test - predictions))
    mae = mae_sum / len(test)

    return rmse, mape, mae


def compute_mse_all_tuned(x_train, y_train, x_test, y_test):
    """Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""

    # define objective function for tuning
    @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
    def tune_cv(x_train, y_train, x_test, y_test, kernel, C, gamma, degree, coef0):
        if kernel == 'linear':
            model = SVR(kernel=kernel, C=C)
        elif kernel == 'poly':
            model = SVR(kernel=kernel, C=C, degree=degree, coef0=coef0)
        elif kernel == 'rbf':
            model = SVR(kernel=kernel, C=C, gamma=gamma)
        else:
            raise ArgumentError("Unknown kernel function: %s" % kernel)
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)

    # optimize parameters
    optimal_pars, _, _ = optunity.minimize_structured(tune_cv, num_evals=1, search_space=space)

    # remove hyperparameters with None value from optimal pars
    for k, v in optimal_pars.items():
        if v is None: del optimal_pars[k]
    print("optimal hyperparameters: " + str(optimal_pars))

    tuned_model = SVR(**optimal_pars).fit(x_train, y_train)
    predictions = tuned_model.predict(x_test)
    return predictions


# def svr_prediction(SR436_Selected_Segments, y_label, train_test_split):
#
#     num_sample_split = int(len(y_label) * train_test_split)
#     # split train and test dataset by 80:20
#     y_test = y_label.iloc[num_sample_split:, :].reset_index(drop=True)
#     test_time = y_test.loc[:, ['Time']]
#
#     total_test = list()
#     total_predict = list()
#     performance = []
#     for inner_outer in range(2):
#         segments = SR436_Selected_Segments[SR436_Selected_Segments['inner_outer'] == inner_outer].sort_values(by=['ES_seq']).reset_index(drop=True)
#         for i in range(len(segments)):
#             pair_id = segments.loc[i, 'Pair_ID']
#             temp = load_file('Data/split_data/segment_data_{}_{}_{}.csv'.format(inner_outer, i, pair_id))
#             # variable selection
#             temp = temp.loc[:, temp.columns.str.contains('|'.join(['avg_speed', 'avg_travel_time']))]
#             print(temp.shape, y_label.shape)
#             # split train and test dataset by 80:20
#             num_sample_split = int(len(temp) * train_test_split)
#             x_train = temp.iloc[:num_sample_split, :].reset_index(drop=True).values
#             x_test = temp.iloc[num_sample_split:, :].reset_index(drop=True).values
#             print(x_train.shape, x_test.shape)
#
#             temp_y = y_label.loc[:, y_label.columns.str.contains('_{}_{}'.format(inner_outer, i))].reset_index(drop=True)
#             y_train = temp_y.iloc[:num_sample_split, :]['y_{}_{}'.format(inner_outer, i)].values
#             y_test = temp_y.iloc[num_sample_split:, :]['y_{}_{}'.format(inner_outer, i)].values
#             print(y_train.shape, y_test.shape)
#
#             # random search
#             estimator = SVR(epsilon=0.01)
#
#             random_search = RandomizedSearchCV(
#                 estimator=estimator,
#                 param_distributions=parameters,
#                 n_iter=5,
#                 scoring='neg_mean_absolute_error',
#                 n_jobs=-1,
#                 cv=3,
#                 verbose=1
#             )
#
#             random_search.fit(x_train, y_train)
#
#             predictions = random_search.best_estimator_.predict(x_test)
#             rmse, mape, mae = evaluation_metrics(y_test, predictions)
#
#             # outer_cv = optunity.cross_validated(x=x_train, y=y_train, num_folds=5)
#             # compute_mse_all_tuned = outer_cv(compute_mse_all_tuned)
#             #
#             # predictions = compute_mse_all_tuned(x_train, y_train, x_test, y_test)
#             # rmse, mape, mae = evaluation_metrics(y_test, predictions)
#             #
#             print(inner_outer, i, rmse, mape, mae)
#             performance.append({'inner_outer': inner_outer, 'i': i, 'mae': mae, 'mape': mape, 'rmse': rmse})
#
#             total_predict.append(predictions)
#             total_test.append(y_test)
#
#             test_prediction = pd.DataFrame({'Time': test_time.Time, 'y_{}_{}_test'.format(inner_outer, i): y_test, 'y_{}_{}_pred'.format(inner_outer, i): predictions})
#             test_time = test_time.join(test_prediction.set_index('Time'), on='Time')
#
#     total_test = np.asarray(total_test).ravel()
#     total_predict = np.asarray(total_predict).ravel()
#     print(total_predict.shape, total_test.shape)
#
#     overall_rmse, overall_mape, overall_mae = evaluation_metrics(total_test, total_predict)
#     performance.append({'inner_outer': 3, 'i': 4, 'mae': overall_mae, 'mape': overall_mape, 'rmse': overall_rmse})
#
#     print(overall_rmse, overall_mape, overall_mae)
#     performance = pd.DataFrame(performance)
#
#     return performance, test_time


def svr_prediction(test_time, parameters):

    x_train, y_train, x_test, y_test = load_data_for_convLSTM()
    num_samples = y_test.shape[0]
    print(num_samples)

    x_train = x_reshape(x_train)
    x_test = x_reshape(x_test)
    y_train = label_reshape(y_train)
    y_test = label_reshape(y_test)

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    # grid search
    # estimator = SVR(epsilon=0.01)
    #
    # random_search = RandomizedSearchCV(
    #     estimator=estimator,
    #     param_distributions=parameters,
    #     n_iter=5,
    #     scoring='neg_mean_absolute_error',
    #     n_jobs=-1,
    #     cv=3,
    #     verbose=1
    # )
    #
    # random_search.fit(x_train, y_train)
    # y_pred = random_search.best_estimator_.predict(x_test)
    clf = SVR(kernel='linear', C=100, verbose=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    y_pred_r = label_reverse(y_pred, num_samples)
    y_test_r = label_reverse(y_test, num_samples)

    print(y_test_r.shape, y_pred_r.shape)

    rmse, mape, mae = evaluation_metrics(y_test_r, y_pred_r)
    print(rmse, mape, mae)

    performance = []
    performance.append({'inner_outer': 3, 'i': 4, 'mae': mae, 'mape': mape, 'rmse': rmse})

    # get the performance for every segment, as well as the final prediction dataset
    y_test_r = y_test_r.reshape(num_samples, 4, 2)
    y_pred_r = y_pred_r.reshape(num_samples, 4, 2)
    for inner_outer in range(2):
        for i in range(4):
            test = y_test_r[:, i, inner_outer]
            predict = y_pred_r[:, i, inner_outer]
            rmse, mape, mae = evaluation_metrics(test, predict)
            print(inner_outer, i, rmse, mape, mae)
            performance.append({'inner_outer': inner_outer, 'i': i, 'mae': mae, 'mape': mape, 'rmse': rmse})

            # final prediction dataset
            prediction = pd.DataFrame({'Time': test_time.Time, 'y_{}_{}_test'.format(inner_outer, i): y_test_r[:, i, inner_outer], 'y_{}_{}_pred'.format(inner_outer, i): y_pred_r[:, i, inner_outer]})
            test_time = test_time.join(prediction.set_index('Time'), on='Time')
            print(inner_outer, i)

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


# space = {'kernel': {'linear': {'C': [0, 100]},
#                     'rbf': {'gamma': [0, 5], 'C': [1, 100]},
#                     'poly': {'degree': [2, 5], 'C': [1000, 20000], 'coef0': [0, 1]}
#                     }
#          }
parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 0.01, 0.1, 0.5, 0.9], 'C': [1, 10, 100]}]

# SR436_Selected_Segments = pd.read_csv('Data/SR436_Selected_Segments.csv')
# intersection_approaches_dict = pd.read_csv('Data/intersection_approaches_dict.csv')
# SR436_Selected_Segments = pd.merge(SR436_Selected_Segments, intersection_approaches_dict[['Seg_ID', 'ES_seq', 'inner_outer']], how='left', left_on=['Seg_ID'], right_on=['Seg_ID'])

y_label = pd.read_csv('Data/split_data/10_min/y_segment_travel_time.csv', index_col=0)
y_label['Time'] = pd.to_datetime(y_label['Time'])
num_sample_split = int(len(y_label) * 0.8)
# split train and test dataset by 80:20
y_test = y_label.iloc[num_sample_split:, :].reset_index(drop=True)
test_time = y_test.loc[:, ['Time']]

# inner_outer = 0
# i = 0
# train_test_split = 0.8
if __name__ == '__main__':
    performance, prediction = svr_prediction(test_time, parameters)
    performance.to_csv('Data/Save_models/SVR/svr_performance.csv', sep=',')
    prediction.to_csv('Data/Save_models/SVR/svr_prediction.csv', sep=',')

    plot = prediction[(prediction['Time'] >= '2018-11-19 00:00:00') & (prediction['Time'] <= '2018-11-20 00:00:00')].reset_index(drop=True)
    prediction_plot(plot, 'Data/Save_models/SVR')

