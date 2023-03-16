"""
Modified on Fri Jan 17 20:01:56 2020

@author: ji758507

Given the raw data, plot occupancy plot for every intersection.
"""

# import all the required packages
import pandas as pd
from datetime import timedelta
from datetime import datetime
import matplotlib.pyplot as plt
import os # For I/O


def get_detection_signal_log(signal_id):

    raw = pd.read_csv('./Data/Raw_Data/Test/Raw_{}.csv'.format(signal_id), skiprows=[0], names=['signal_id', 'timestamp', 'event_code', 'event_parameter'])
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    raw = raw.sort_values(by=['timestamp'])

    detection = raw[raw['event_code'].isin([81, 82])]
    detection = detection.sort_values(by=['event_parameter', 'timestamp'])

    signal = raw[raw['event_code'].isin([1, 8, 10])]
    signal = signal.sort_values(by=['event_parameter', 'timestamp'])

    return detection, signal


# define the function to plot the configuration of selected detectors
def get_detection(signal_id, detection, signal, crash_time, window):
    start_time = crash_time - timedelta(minutes=window)
    end_time = crash_time + timedelta(minutes=window)

    channel = detector_list.loc[detector_list['SignalID'] == signal_id, 'Det_Channel'].tolist()
    detection_movement = detection[detection['event_parameter'].isin(channel)]
    detection_crash_time = detection_movement[(detection_movement['timestamp'] > start_time) & (detection_movement['timestamp'] < end_time)].reset_index(drop=True)

    phase = detector_list.loc[detector_list['SignalID'] == signal_id, 'Phase'].unique().tolist()
    signal_movement = signal[signal['event_parameter'].isin(phase)]
    signal_crash_time = signal_movement[(signal_movement['timestamp'] > start_time) & (signal_movement['timestamp'] < end_time)].reset_index(drop=True)

    return detection_crash_time, signal_crash_time


# def get_occupancy(detection_channel):
#
#     detector_on = detection_channel[detection_channel['event_code'] == 82].reset_index(drop=True)
#     detector_off = detection_channel[detection_channel['event_code'] == 81].reset_index(drop=True)
#     if detector_on.loc[0, 'timestamp'] > detector_off.loc[0, 'timestamp']:
#         detector_off = detector_off.iloc[1:, ].reset_index(drop=True)
#
#     occupancy = pd.concat([detector_on, detector_off], ignore_index=True, axis=1)
#     occupancy = occupancy.loc[(occupancy[1].notnull()) & (occupancy[5].notnull())]
#     occupancy['occupancy'] = abs((occupancy[5] - occupancy[1]).dt.total_seconds())
#
#     return occupancy


def get_occupancy_headway(channel_detection):
    # delete error data (event_code = lead_event_code = 81 or event_code = lag_event_code = 82)
    channel_detection['lag_event_code'] = channel_detection['event_code'].shift(-1)
    channel_detection['lead_event_code'] = channel_detection['event_code'].shift(1)
    channel_detection = channel_detection[~(((channel_detection['event_code'] == 81) & (channel_detection['lead_event_code'] == 81)) | ((channel_detection['event_code'] == 82) & (channel_detection['lag_event_code'] == 82)))][['signal_id', 'timestamp', 'event_code', 'event_parameter']].reset_index(drop=True)

    channel_detection['lag_event_code'] = channel_detection['event_code'].shift(-1)
    channel_detection['lag_timestamp'] = channel_detection['timestamp'].shift(-1)
    channel_detection['lead_event_code'] = channel_detection['event_code'].shift(1)
    channel_detection['lead_timestamp'] = channel_detection['timestamp'].shift(1)

    channel_detection = channel_detection[(channel_detection['event_code'] == 82) & (channel_detection['lag_event_code'].notnull()) & (channel_detection['lead_event_code'].notnull())].reset_index(drop=True)

    channel_detection['next_detec_timestamp'] = channel_detection['timestamp'].shift(-1)
    channel_detection['occupancy'] = (channel_detection['lag_timestamp'] - channel_detection['timestamp']).dt.total_seconds()
    channel_detection['gap'] = (channel_detection['timestamp'] - channel_detection['lead_timestamp']).dt.total_seconds()
    channel_detection['headway'] = channel_detection['occupancy'] + channel_detection['gap']
    channel_detection['lag_gap'] = (channel_detection['next_detec_timestamp'] - channel_detection['lag_timestamp']).dt.total_seconds()

    # filter out error data
    channel_detection = channel_detection[(channel_detection['occupancy'] > 0) & (channel_detection['gap'] > 0) & (channel_detection['occupancy'] < 3600) & (channel_detection['headway'] < 3600) & (channel_detection['lag_gap'].notnull())].reset_index(drop=True)
    return channel_detection


def main(selected_signal_list, detector_list, window):
    # for i in range(411, 641):
    for i in range(len(selected_signal_list)):
        crash_time = datetime.strptime('2019-09-04 12:00:00', '%Y-%m-%d %H:%M:%S')
        signal_id = selected_signal_list[i]

        detection, signal = get_detection_signal_log(signal_id)
        detection_crash_time, signal_crash_time = get_detection(signal_id, detection, signal, crash_time, window)

        if len(signal_crash_time) > 0 and len(detection_crash_time) > 0:

            # phase_list = signal_crash_time['event_parameter'].unique().tolist()
            channel_list = detection_crash_time['event_parameter'].unique().tolist()
            num_channels = len(channel_list)

            detection_table = detector_list.loc[(detector_list['Det_Channel'].isin(channel_list)) & (detector_list['SignalID'] == signal_id)]
            detection_table = detection_table.sort_values(by=['Phase', 'Det_Channel']).reset_index(drop=True)

            fig = plt.figure(figsize=(40, 80))
            plt.rcParams.update({'font.size': 16})

            for j in range(len(detection_table)):
                channel = detection_table.loc[j, 'Det_Channel']
                phase = detection_table.loc[j, 'Phase']
                phase_movement = detection_table.loc[(detection_table['Phase'] == phase) & (detection_table['SignalID'] == signal_id), 'Movement'].tolist()[0]

                signal_log = signal_crash_time[signal_crash_time['event_parameter'] == phase].reset_index(drop=True)
                signal_log['lag_event_code'] = signal_log['event_code'].shift(-1)
                signal_log['lag_timestamp'] = signal_log['timestamp'].shift(-1)

                channel_detection = detection_crash_time[detection_crash_time['event_parameter'] == channel].reset_index(drop=True)
                channel_occupancy = get_occupancy_headway(channel_detection)

                ax = fig.add_subplot(num_channels, 1, j+1)

                start_time = crash_time - timedelta(minutes=window)
                end_time = crash_time + timedelta(minutes=window)

                ts = pd.date_range(start=start_time, end=end_time, freq='T')
                ax.plot_date(channel_occupancy['timestamp'], channel_occupancy['occupancy'], color='black')
                plt.xticks(ts)

                for k in range(len(signal_log)):
                    if signal_log.loc[k, 'event_code'] == 1 and signal_log.loc[k, 'lag_event_code'] == 8:
                        green_start = signal_log.loc[k, 'timestamp']
                        green_end = signal_log.loc[k, 'lag_timestamp']
                        plt.axvspan(green_start, green_end, facecolor='#00b04e', alpha=1)

                    if signal_log.loc[k, 'event_code'] == 8 and signal_log.loc[k, 'lag_event_code'] == 10:
                        yellow_start = signal_log.loc[k, 'timestamp']
                        yellow_end = signal_log.loc[k, 'lag_timestamp']
                        plt.axvspan(yellow_start, yellow_end, facecolor='#ffff3d', alpha=0.8)

                    if signal_log.loc[k, 'event_code'] == 10 and signal_log.loc[k, 'lag_event_code'] == 1:
                        red_start = signal_log.loc[k, 'timestamp']
                        red_end = signal_log.loc[k, 'lag_timestamp']
                        plt.axvspan(red_start, red_end, facecolor='#f04044', alpha=0.8)

                    if signal_log.loc[k, 'event_code'] == 1 and signal_log.loc[k, 'lag_event_code'] == 10:
                        green_start = signal_log.loc[k, 'timestamp']
                        green_end = signal_log.loc[k, 'lag_timestamp']
                        plt.axvspan(green_start, green_end, facecolor='#00b04e', alpha=1)

                    if signal_log.loc[k, 'event_code'] == 1 and signal_log.loc[k, 'lag_event_code'] == 1:
                        green_start = signal_log.loc[k, 'timestamp']
                        green_end = signal_log.loc[k, 'lag_timestamp']
                        plt.axvspan(green_start, green_end, facecolor='#00b04e', alpha=1)

                    if signal_log.loc[k, 'event_code'] == 8 and signal_log.loc[k, 'lag_event_code'] == 1:
                        yellow_start = signal_log.loc[k, 'timestamp']
                        yellow_end = signal_log.loc[k, 'lag_timestamp']
                        plt.axvspan(yellow_start, yellow_end, facecolor='#ffff3d', alpha=0.8)

                    if signal_log.loc[k, 'event_code'] == 10 and signal_log.loc[k, 'lag_event_code'] == 8:
                        red_start = signal_log.loc[k, 'timestamp']
                        red_end = signal_log.loc[k, 'lag_timestamp']
                        plt.axvspan(red_start, red_end, facecolor='#f04044', alpha=0.8)

                    print(signal_id, j, k)

                labels = list(range(-window, (window + 1)))
                ax.set_xticklabels(labels)
                ax.set_xlabel('Time (minutes)')
                ax.set_ylabel('Occupancy (second)')
                ax.set_title('Occupancy of ' + str(phase_movement) + ' at Channel ' + str(channel))

            fig.savefig('./Data/Occupancy_Plot/signal_{}_occupancy.png'.format(signal_id), dpi=300)


# load ATSPM detector configuration and ATSPM intersection
detector_list = pd.read_csv('./Data/Detector_Checklist_final.csv')
# atspm_intersections = pd.read_excel(r'J:/ATSPM_CNN_LSTM/Data/Signal Location-Revise.xlsx', sheet_name='Queue')
# extract the list of intersections based on what raw data we have in the folder
path = os.getcwd()
data_path = os.path.join(path, 'Data\Raw_Data\Test')
files = os.listdir(data_path) # Get files path within the folder
intersections = [int(f[4:8]) for f in files if f[0:3] == 'Raw']

# filter out the approaches with two groups of detectors
# selected_signal_list = atspm_intersections['Signal_Id'].unique().tolist()
# len(selected_signal_list)
# plot_signal_detector_list = detector_list[detector_list['SignalID'].isin(selected_signal_list)].reset_index(drop=True)
# len(plot_signal_detector_list['SignalID'].unique().tolist())
# selected_signal_list = detector_list['SignalID'].unique().tolist()
if __name__ == '__main__':
    main(intersections, detector_list, window=30)

    # raw = pd.read_csv('./Data/Raw_Data/Test/Raw_1230.csv', skiprows=[0], names=['signal_id', 'timestamp', 'event_code', 'event_parameter'])
