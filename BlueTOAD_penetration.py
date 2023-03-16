"""
Created on Fri Jan 17 20:01:56 2020

@author: ji758507
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os
from matplotlib.dates import DateFormatter


root_path = 'Data/split_data/5_min'

model_path = os.path.join(root_path)
files = os.listdir(model_path)

x_data = [i for i in files if 'segment_data' in i]
y_data = [i for i in files if 'y_segment' in i][0]

for inner_outer in range(2):

    for segment in range(4):

        x_file = [i for i in x_data if '_{}_{}_'.format(inner_outer, segment) in i][0]

        x_tem = pd.read_csv(os.path.join(root_path + '/' + x_file), index_col=0)

        x_tem['Time'] = pd.to_datetime(x_tem['Time'])

        penetration_ratio = sum(x_tem['count_{}_{}_1'.format(inner_outer, segment)])/sum(x_tem['cycle_volume_Down_{}_{}_1'.format(inner_outer, segment)])

        y_tem = pd.read_csv(os.path.join(root_path + '/' + y_data), index_col=0)
        y_tem['Time'] = pd.to_datetime(y_tem['Time'])


def comparison_plot(root_path, y_tem, start_time, end_time):

    y_tem = y_tem[(y_tem['Time'] >= start_time) & (y_tem['Time'] <= end_time)].reset_index(drop=True)

    fig = plt.figure(figsize=(60, 50))
    plt.rcParams.update({'font.size': 30})
    formatter = DateFormatter('%Y-%m-%d')

    for i in range(4):
        for inner_outer in range(2):

            index = 2 * i + 1 + inner_outer
            ax = fig.add_subplot(4, 2, index)

            ax.plot_date(y_tem['Time'], y_tem['y_{}_{}'.format(inner_outer, i)], color='black', alpha=0.7, fmt='b-', linewidth=2.0, label='Travel Time')
            print(i, inner_outer)

            ts = pd.date_range(start=start_time, end=end_time, freq='1440T')
            ax.set_xticks(ts)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_tick_params(rotation=0, labelsize=30)
            ax.set_xlabel('Time')
            ax.set_ylabel('Travel Time (s)')
            ax.set_ylim(0, 100)
            ax.legend(loc="upper right")

            if inner_outer == 0:
                direction = 'Eastbound'
            else:
                direction = 'Westbound'
            ax.set_title('Segment ID: {} ({})'.format(index, direction))

    fig.savefig('Data/Save_models/travel_time.png', transparent=False, dpi=200)


def comparison_plot_one_direction(root_path, y_tem, start_time, end_time):

    y_tem = y_tem[(y_tem['Time'] >= start_time) & (y_tem['Time'] <= end_time)].reset_index(drop=True)

    fig = plt.figure(figsize=(50, 60))
    plt.rcParams.update({'font.size': 40})
    formatter = DateFormatter('%Y-%m-%d')

    for i in range(4):
        # for inner_outer in range(2):
        inner_outer = 0
        index = i + 1
        ax = fig.add_subplot(4, 1, index)

        ax.plot_date(y_tem['Time'], y_tem['y_{}_{}'.format(inner_outer, i)], color='black', alpha=0.7, fmt='b-', linewidth=2.0, label='Travel Time')
        print(i, inner_outer)

        ts = pd.date_range(start=start_time, end=end_time, freq='1440T')
        ax.set_xticks(ts)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(rotation=0, labelsize=40)
        ax.set_xlabel('Time')
        ax.set_ylabel('Travel Time (s)')
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right")

        if inner_outer == 0:
            direction = 'Eastbound'
        else:
            direction = 'Westbound'
        ax.set_title('Segment ID: {} ({})'.format(index, direction))

        for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
            ax.spines[side].set_linewidth(3)

    fig.savefig('Data/Save_models/travel_time.png', transparent=False, dpi=200)


inner_outer = 0
segment = 0

start_time = '2018-10-10 00:00:00'
end_time = '2018-10-17 00:00:00'
comparison_plot_one_direction(root_path, y_tem, start_time, end_time)
