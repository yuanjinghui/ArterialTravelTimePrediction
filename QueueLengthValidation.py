"""
Created on Fri Jan 17 20:01:56 2020

@author: ji758507
"""

import pandas as pd
import matplotlib.pyplot as plt
import math


raw_data = pd.read_csv('./Data/raw_data.csv')
ground_truth = pd.read_excel('Data/VideoForQueueLengthValidation.xlsx')

ground_truth = pd.merge(ground_truth, raw_data, how='left', left_on=['signal_id', 'cycle_num', 'movement'], right_on=['signal_id', 'cycle_num', 'movement'])
ground_truth = ground_truth.fillna(0)

# evaluation performance metrics
ground_truth['queue_error'] = abs(ground_truth['max_queue_length'] - ground_truth['GroundTruthMaxQueue'])
ground_truth['queue_error_percent'] = abs(ground_truth['max_queue_length'] - ground_truth['GroundTruthMaxQueue'])/ground_truth['GroundTruthMaxQueue']
ground_truth['queue_error_square'] = abs(ground_truth['max_queue_length'] - ground_truth['GroundTruthMaxQueue'])**2
Mae = ground_truth['queue_error'].sum()/len(ground_truth)
Mape = ground_truth['queue_error_percent'].sum()/len(ground_truth)
Rmse = math.sqrt(ground_truth['queue_error_square'].sum()/len(ground_truth))

# plot
fig2 = plt.figure(figsize=(20, 8))
plt.rcParams.update({'font.size': 15})

ax1 = fig2.add_subplot(1, 1, 1)
# set width of bar
barWidth = 0.3

temp_plot = ground_truth[['GroundTruthMaxQueue', 'max_queue_length']]
# set height of bar
# bars1 = temp.loc[temp['model_type'] == 'ConvLSTM', ['sensitivity', 'false_AR']]
# bars2 = temp.loc[temp['model_type'] == 'CNN_LSTM', ['sensitivity', 'false_AR']])
# bars3 = list(temp.loc[temp['model_type'] == 'LSTM', ['sensitivity', 'false_AR']])

# Set position of bar on X axis
r1 = list(range(len(temp_plot['max_queue_length'])))
r2 = [x + barWidth for x in r1]

# Make the plot
rects1 = ax1.bar(r1, temp_plot['GroundTruthMaxQueue'], color='darkorange', width=barWidth, edgecolor='white', label='Ground Truth')
rects2 = ax1.bar(r2, temp_plot['max_queue_length'], color='darkgreen', width=barWidth, edgecolor='white', label='Predicted')

# Add xticks on the middle of the group bars
# ax1.set_xlabel('Classification Metric', fontweight='bold')
# ax.set_xticks([p + 1.5 * width for p in pos])

for rect in rects1:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
             '%d' % int(height),
             ha='center', va='bottom')

for rect in rects2:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
             '%d' % int(height),
             ha='center', va='bottom')

ax1.set_ylim([0, 25])

ax1.set_xticks([r + barWidth for r in r1])
ax1.set_xticklabels(temp_plot.index)
ax1.set_title('Validation of Queue Length Estimation', fontweight='bold')
ax1.set_ylabel('Maximum Queue Length (veh)')
ax1.set_xlabel('Cycles')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = '#88cccc'
ax2.set_ylabel('Queue Over Detector (0/1)')  # we already handled the x-label with ax1
ax2.plot(ground_truth['A_exist'], linestyle='solid', color=color, linewidth=3.0, label='Queue Over Detector')
ax2.set_ylim([0, 2])
ax2.set_yticks([0, 1])
# Create legend & Show graphic
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()