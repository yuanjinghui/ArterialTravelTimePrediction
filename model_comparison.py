"""
Created on Mon Feb 12 20:01:56 2020

@author: ji758507
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def load_model_prediction(root_path, model_type, prediction_horizon, start_time, end_time):

    if model_type in (['ConvLSTM', 'CNN_LSTM', 'LSTM', 'GRU']):
        folder_path = os.path.join(root_path + model_type + '/' + 'All_variables_server_{}_min'.format(prediction_horizon))
        model_tuning_performance = pd.read_csv(folder_path + '/' + '{}_tuning_performance.csv'.format(model_type), index_col=0)
        best_model_id = model_tuning_performance.loc[model_tuning_performance['rank'] == 1]['sub_model_id'][0]
        model_predictions = pd.read_csv(os.path.join(folder_path + '/' + 'Prediction_performance_{}_{}.csv'.format(model_type, best_model_id)), index_col=0)

    else:
        folder_path = os.path.join(root_path + model_type + '/' + '{}_min'.format(prediction_horizon))
        model_predictions = pd.read_csv(os.path.join(folder_path + '/' + '{}_prediction.csv'.format(model_type.lower())), index_col=0)

    model_predictions['Time'] = pd.to_datetime(model_predictions['Time'])
    prediction_plot = model_predictions[(model_predictions['Time'] >= start_time) & (model_predictions['Time'] <= end_time)].reset_index(drop=True)

    return prediction_plot


def comparison_plot(root_path, model_list, prediction_horizon, start_time, end_time, fig_name):

    fig = plt.figure(figsize=(50, 60))
    plt.rcParams.update({'font.size': 35})
    formatter = DateFormatter('%H:%M')

    for i in range(4):
        for inner_outer in range(2):

            index = 2 * i + 1 + inner_outer
            ax = fig.add_subplot(4, 2, index)

            for j in range(len(model_list)):
                model_type = model_list[j]
                prediction_plot = load_model_prediction(root_path, model_type, prediction_horizon, start_time, end_time)
                test = pd.concat([prediction_plot.Time, prediction_plot.iloc[:, prediction_plot.columns.str.contains('_{}_{}'.format(inner_outer, i))]], axis=1)

                if j == 0:
                    # test = test.set_index('Time')
                    ax.plot_date(test['Time'], test['y_{}_{}_test'.format(inner_outer, i)], color='black', fmt='b-', linewidth=3.0, label='Ground Truth')
                    ax.plot_date(test['Time'], test['y_{}_{}_pred'.format(inner_outer, i)], color='C{}'.format(j), alpha=0.8, fmt='b--', linewidth=3.0, label=model_type)
                else:
                    ax.plot_date(test['Time'], test['y_{}_{}_pred'.format(inner_outer, i)], color='C{}'.format(j), alpha=0.8, fmt='b--', linewidth=3.0, label=model_type)
                print(i, inner_outer, j)
            ts = pd.date_range(start=start_time, end=end_time, freq='60T')
            ax.set_xticks(ts)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_tick_params(rotation=0, labelsize=35)
            # ax.set_ylim(0, 100)
            ax.set_xlabel('Time')
            ax.set_ylabel('Travel Time (s)')
            ax.legend(loc="upper left")

            if inner_outer == 0:
                direction = 'Eastbound'
            else:
                direction = 'Westbound'
            ax.set_title('{} Minutes Prediction for Segment ID: {} ({})'.format(prediction_horizon, index, direction))

            for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
                ax.spines[side].set_linewidth(3)

    fig.savefig(root_path + '/' + '/Comparison_{}_{}.png'.format(fig_name, prediction_horizon), transparent=False, dpi=200)


model_list = ['HA', 'ARIMA', 'ConvLSTM', 'LSTM', 'GRU']
# model_list = ['ConvLSTM', 'CNN_LSTM', 'LSTM', 'GRU', 'XGB', 'ARIMA', 'HA']
# model_list = ['CNN_LSTM', 'LSTM', 'GRU', 'XGB', 'ARIMA', 'HA']
root_path = 'Data/Save_models/'
# prediction_horizon = 5
# start_time = '2018-11-19 13:00:00'
# end_time = '2018-11-19 17:00:00'
# weekday (Friday)
start_time = '2018-11-19 07:00:00'
end_time = '2018-11-19 10:00:00'
fig_name = 'weekday'
# weekend (Saturday)
# start_time = '2018-11-24 00:00:00'
# end_time = '2018-11-25 00:00:00'
# fig_name = 'weekend'
# model_type = 'ConvLSTM'
if __name__ == '__main__':
    for prediction_horizon in ([5, 10, 15, 20, 25, 30]):

        comparison_plot(root_path, model_list, prediction_horizon, start_time, end_time, fig_name)






