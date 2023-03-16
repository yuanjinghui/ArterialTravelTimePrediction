"""
Created on Mon Jul 21 20:01:56 2019

@author: ji758507
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# ==========================================================================================================
# Historical Average (same time of the day and day of the week)
# ==========================================================================================================
def evaluation_metrics(test, predictions):
    mse_sum = sum((test - predictions) ** 2)
    rmse = math.sqrt(mse_sum / len(test))

    mape_sum = sum(abs(test - predictions) / test)
    mape = mape_sum / len(test)

    mae_sum = sum(abs(test - predictions))
    mae = mae_sum / len(test)

    return rmse, mape, mae


def historical_average(y_label, train_test_split):
    num_sample_split = int(len(y_label) * train_test_split)
    train = y_label.iloc[:num_sample_split, :].reset_index(drop=True)
    test = y_label.iloc[num_sample_split:, :].reset_index(drop=True)

    test['date'] = test['Time'].dt.date
    date_list = test['date'].unique().tolist()
    prediction = []
    for i in date_list:
        tem = test[test['date'] == i].reset_index(drop=True)

        tem = pd.merge(tem, train, how='left', left_on=['time_of_day', 'day_of_week'], right_on=['time_of_day', 'day_of_week'], suffixes=('_test', '_pred'), validate='one_to_many')
        tem = tem.groupby(by=['Time_test'], as_index=False).mean()
        prediction.append(tem)
        # prediction = prediction.join(tem.set_index('Time_test'), on='Time')
        print(i)
    prediction = pd.concat(prediction, axis=0)
    prediction.rename(columns={'Time_test': 'Time'}, inplace=True)
    return prediction


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
    fig.savefig(root_path + '/Comparison.png', transparent=False, dpi=400)


y_label = pd.read_csv('Data/split_data/15_min/y_segment_travel_time.csv', index_col=0)
y_label.Time = pd.to_datetime(y_label.Time)
y_label['time_of_day'] = y_label['Time'].dt.time
y_label['day_of_week'] = y_label['Time'].dt.dayofweek

train_test_split = 0.8
# i = datetime.date(2018, 11, 18)
if __name__ == '__main__':
    prediction = historical_average(y_label, train_test_split)
    performance = evaluation(prediction)
    performance.to_csv('Data/Save_models/HA/HA_performance.csv', sep=',')
    prediction.to_csv('Data/Save_models/HA/HA_prediction.csv', sep=',')
    plot = prediction[(prediction['Time'] >= '2018-11-19 00:00:00') & (prediction['Time'] <= '2018-11-20 00:00:00')].reset_index(drop=True)
    prediction_plot(plot, 'Data/Save_models/HA')


# inner_outer = 0
# i = 1
