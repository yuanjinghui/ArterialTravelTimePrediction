"""
Modified on Fri Jan 17 20:01:56 2020

@author: ji758507

Calculate the Max_AOR to validate the distance between detector and stop bar

"""
import pandas as pd
import os # For I/O


def load_detection_signal_log(signal_id):
    """
    Load vehicle detection and signal log data for the given intersection id.
    :param signal_id: signal id
    :return: two dataframes include vehicle actuation and signal log information, respeectively.
    """
    raw = pd.read_csv('./Data/Raw_Data/Test/Raw_{}.csv'.format(signal_id), skiprows=[0], names=['signal_id', 'timestamp', 'event_code', 'event_parameter'])
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    raw = raw.sort_values(by=['timestamp'])

    detection = raw[raw['event_code'].isin([81, 82])]
    detection = detection.sort_values(by=['event_parameter', 'timestamp'])
    # signal = raw[raw['event_code'].isin(list(range(1, 12)))]
    signal = raw[raw['event_code'].isin([1, 8, 10])]
    signal = signal.sort_values(by=['event_parameter', 'timestamp'])

    return detection, signal


def clean_detector_config(detector_config, atspm_intersections):
    """
    Clean the original detector configuration table
    :param detector_config: original detector configuration table
    :param atspm_intersections: ATSPM intersection dictionary
    :return: selected_intersection_list and save cleaned configuration table to local csv file
    """
    # extract all the available detector channels
    detector_config_r = detector_config[detector_config['Phase'].notnull() & detector_config['Volume'].isin([1, 2, 3])].reset_index(drop=True)
    detector_config_r.loc[detector_config_r['Volume_Movement'].isnull(), 'Volume_Movement'] = detector_config_r['Movement']
    detector_config_r.Phase = detector_config_r.Phase.astype(int)
    detector_config_r = detector_config_r.fillna(999)

    # exclude all the non-selected intersections
    # selected_intersection_list = list(atspm_intersections.loc[atspm_intersections['Selected'] == 1, 'Signal_Id'])
    detector_config_r = detector_config_r[detector_config_r['SignalID'].isin(atspm_intersections)].reset_index(drop=True)

    # save to csv file
    detector_config_r.to_csv('./TestDetectorConfig.csv', sep=',')

    return detector_config_r


def channel_volume_max_aor(intersection_list, detector_config_r, start_time, end_time):
    """
    Generate the total volume and maximum arrival on red to verify the
    location and status of every detector channel.
    :param signal_list: list of signals for verification.
    :return: save to a csv file.
    """
    aor_count = []
    total_volume = []
    for id in intersection_list:
        # load the detection and signal timing data for selected intersection
        detection, signal = load_detection_signal_log(id)

        # get the list of phase number for the selected intersection
        phase_list = detector_config_r.loc[detector_config_r['SignalID'] == id, 'Phase'].unique().tolist()

        for phase_id in phase_list:
            detector_table = detector_config_r.loc[(detector_config_r['SignalID'] == id) & (detector_config_r['Phase'] == phase_id)].reset_index(drop=True)
            channel_list = detector_table['Det_Channel'].unique().tolist()

            # extract signal timing data for selected phase id
            phase_signal = signal[signal['event_parameter'] == phase_id].reset_index(drop=True)

            # for continuous repeated event code, delete the following ones
            phase_signal['event_code_diff'] = phase_signal['event_code'] - phase_signal['event_code'].shift(1)
            phase_signal = phase_signal[phase_signal['event_code_diff'] != 0].reset_index(drop=True)
            phase_signal = phase_signal.drop(['event_code_diff'], axis=1)

            # generate four lag event code
            phase_signal['lag_event_code'] = phase_signal['event_code'].shift(-1)
            phase_signal['lag_timestamp'] = phase_signal['timestamp'].shift(-1)

            phase_signal['lag2_event_code'] = phase_signal['event_code'].shift(-2)
            phase_signal['lag2_timestamp'] = phase_signal['timestamp'].shift(-2)

            phase_signal['lag3_event_code'] = phase_signal['event_code'].shift(-3)
            phase_signal['lag3_timestamp'] = phase_signal['timestamp'].shift(-3)

            phase_signal['lag4_event_code'] = phase_signal['event_code'].shift(-4)
            phase_signal['lag4_timestamp'] = phase_signal['timestamp'].shift(-4)

            phase_signal = phase_signal[(phase_signal['event_code'] == 10) & (phase_signal['lag_event_code'].notnull())].reset_index(drop=True)

            # generate the start time and end time of whole cycle, green, yellow, and red
            phase_signal['BOC'] = phase_signal['timestamp']
            phase_signal['EOC'] = phase_signal['timestamp'].shift(-1)
            phase_signal['cycle_len'] = (phase_signal['EOC'] - phase_signal['BOC']).dt.total_seconds()

            # delete the abnormal signal data
            phase_signal = phase_signal[(phase_signal['cycle_len'].notnull()) & (phase_signal['cycle_len'] < 7200) & (phase_signal['cycle_len'] > 10)].reset_index(drop=True)

            phase_signal['BOR'] = phase_signal['timestamp']
            phase_signal['EOR'] = phase_signal['lag_timestamp']

            # list the four major signal timing patterns (99.8833%) and then revised them to be standard version
            # lag0   lag1   lag2   lag3   lag4
            # 10    1   8   10
            # 10    1   10
            # 10    8   1   8   10
            # 10    8   1   10
            # 10    8   10
            phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 8) & (phase_signal['lag3_event_code'] == 10), 'BOG'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 8) & (phase_signal['lag3_event_code'] == 10), 'lag_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 8) & (phase_signal['lag3_event_code'] == 10), 'EOG'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 8) & (phase_signal['lag3_event_code'] == 10), 'lag2_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 8) & (phase_signal['lag3_event_code'] == 10), 'BOY'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 8) & (phase_signal['lag3_event_code'] == 10), 'lag2_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 8) & (phase_signal['lag3_event_code'] == 10), 'EOY'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 8) & (phase_signal['lag3_event_code'] == 10), 'lag3_timestamp']

            phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 10), 'BOG'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 10), 'lag_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 10), 'EOG'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 10), 'lag2_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 10), 'BOY'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 10), 'lag2_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 10), 'EOY'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 10), 'lag2_timestamp']

            phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'BOG'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'lag2_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'EOG'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'lag3_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'BOY'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'lag3_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'EOY'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'lag4_timestamp']

            phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 10), 'BOG'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 10), 'lag2_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 10), 'EOG'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 10), 'lag3_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 10), 'BOY'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 10), 'lag3_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 10), 'EOY'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 10), 'lag3_timestamp']

            phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 10), 'BOG'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 10), 'lag_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 10), 'EOG'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 10), 'lag_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 10), 'BOY'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 10), 'lag_timestamp']
            phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 10), 'EOY'] \
                = phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 10), 'lag2_timestamp']

            # keep only the abovementioned five signal patterns
            phase_signal = phase_signal[((phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 8) & (phase_signal['lag3_event_code'] == 10))
                                          | ((phase_signal['lag_event_code'] == 1) & (phase_signal['lag2_event_code'] == 10))
                                          | ((phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10))
                                          | ((phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 10))
                                          | ((phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 10))].reset_index(drop=True)

            # fill in null value of green and yellow phase
            phase_signal.loc[phase_signal['BOG'].isnull(), 'BOG'] = phase_signal.loc[phase_signal['BOG'].isnull(), 'EOR']
            phase_signal.loc[phase_signal['EOG'].isnull(), 'EOG'] = phase_signal.loc[phase_signal['EOG'].isnull(), 'BOG']

            phase_signal.loc[phase_signal['BOY'].isnull(), 'BOY'] = phase_signal.loc[phase_signal['BOY'].isnull(), 'EOG']
            phase_signal.loc[phase_signal['EOY'].isnull(), 'EOY'] = phase_signal.loc[phase_signal['EOY'].isnull(), 'BOY']

            # calculate green time, yellow time, and red time
            phase_signal['red_time'] = (phase_signal['EOR'] - phase_signal['BOR']).dt.total_seconds()
            phase_signal['green_time'] = (phase_signal['EOG'] - phase_signal['BOG']).dt.total_seconds()
            phase_signal['yellow_time'] = (phase_signal['EOY'] - phase_signal['BOY']).dt.total_seconds()

            phase_signal['cycle_num'] = phase_signal.index + 1

            # detection data process
            temp = []
            for j in channel_list:
                channel_detection = detection[detection['event_parameter'] == j].reset_index(drop=True)

                # delete error data (event_code = lead_event_code = 81 or event_code = lag_event_code = 82)
                channel_detection['lag_event_code'] = channel_detection['event_code'].shift(-1)
                channel_detection['lead_event_code'] = channel_detection['event_code'].shift(1)
                channel_detection = channel_detection[~(((channel_detection['event_code'] == 81) & (channel_detection['lead_event_code'] == 81)) | (
                        (channel_detection['event_code'] == 82) & (channel_detection['lag_event_code'] == 82)))][['signal_id', 'timestamp', 'event_code', 'event_parameter', 'lag_event_code', 'lead_event_code']].reset_index(drop=True)

                channel_detection['lag_timestamp'] = channel_detection['timestamp'].shift(-1)
                channel_detection['lead_timestamp'] = channel_detection['timestamp'].shift(1)

                channel_detection = channel_detection[
                    (channel_detection['event_code'] == 82) & (channel_detection['lag_event_code'].notnull()) & (channel_detection['lead_event_code'].notnull())].reset_index(drop=True)

                channel_detection['occupancy'] = (channel_detection['lag_timestamp'] - channel_detection['timestamp']).dt.total_seconds()
                channel_detection['gap'] = (channel_detection['timestamp'] - channel_detection['lag_timestamp'].shift(1)).dt.total_seconds()
                channel_detection['headway'] = channel_detection['occupancy'] + channel_detection['gap']

                # filter out error data
                channel_detection = channel_detection[(channel_detection['occupancy'] < 3600)].reset_index(drop=True)

                # identify the cycle number and green, yellow, red identification
                cycle_index = pd.IntervalIndex.from_arrays(phase_signal['BOC'], phase_signal['EOC'], closed='left')
                channel_detection['cycle_num'] = phase_signal.loc[cycle_index.get_indexer(channel_detection.timestamp), 'cycle_num'].values
                channel_detection = channel_detection[channel_detection['cycle_num'].notnull()].reset_index(drop=True)

                green_index = pd.IntervalIndex.from_arrays(phase_signal['BOG'], phase_signal['EOG'], closed='left')
                channel_detection['green_cycle_num'] = phase_signal.loc[green_index.get_indexer(channel_detection.timestamp), 'cycle_num'].values

                yellow_index = pd.IntervalIndex.from_arrays(phase_signal['BOY'], phase_signal['EOY'], closed='left')
                channel_detection['yellow_cycle_num'] = phase_signal.loc[yellow_index.get_indexer(channel_detection.timestamp), 'cycle_num'].values

                red_index = pd.IntervalIndex.from_arrays(phase_signal['BOR'], phase_signal['EOR'], closed='left')
                channel_detection['red_cycle_num'] = phase_signal.loc[red_index.get_indexer(channel_detection.timestamp), 'cycle_num'].values

                channel_detection.loc[channel_detection['green_cycle_num'].notnull(), 'signal_status'] = 'green'
                channel_detection.loc[channel_detection['yellow_cycle_num'].notnull(), 'signal_status'] = 'yellow'
                channel_detection.loc[channel_detection['red_cycle_num'].notnull(), 'signal_status'] = 'red'

                temp.append(channel_detection)

                print(id, phase_id, j)

            total_channel_detection = pd.concat(temp, ignore_index=True)

            # calculate the AOR for every detection channel on every cycle
            aor = total_channel_detection[total_channel_detection['signal_status'] == 'red'].reset_index(drop=True)
            aor = aor.groupby(by=['signal_id', 'event_parameter', 'cycle_num'], as_index=False).agg({"occupancy": ['max', 'mean', 'std', 'count'], "headway": ['mean', 'std']})
            aor.columns = ['signal_id', 'event_parameter', 'cycle_num', 'max_occupancy', 'avg_occupancy', 'std_occupancy', 'count', 'avg_headway', 'std_headway']
            aor = aor.join(phase_signal[['cycle_num', 'cycle_len', 'timestamp']].set_index('cycle_num'), on='cycle_num')
            # select the morning peak hour data
            aor = aor[(aor['timestamp'] >= start_time) & (aor['timestamp'] <= end_time)].reset_index(drop=True)
            max_aor = aor[['signal_id', 'event_parameter', 'max_occupancy', 'count']].groupby(by=['signal_id', 'event_parameter'], as_index=False).quantile(1, interpolation='lower')
            aor_count.append(max_aor)

            # calculate the total volume of every detector
            volume = total_channel_detection.groupby(by=['signal_id', 'event_parameter'], as_index=False).size().reset_index(name='tot_volume')
            total_volume.append(volume)

    detector_volume = pd.concat(total_volume, ignore_index=True)
    total_aor_count = pd.concat(aor_count, ignore_index=True)
    total_aor_count = pd.merge(total_aor_count, detector_volume, how='left', left_on=['signal_id', 'event_parameter'], right_on=['signal_id', 'event_parameter'])

    return total_aor_count


# id = 1510
# phase_id = 5
# j = 4
# aor['count'].describe()
# aor['count'].max()
# aor['count'].quantile(.95)
# start_time = '2017-05-03 09:00:00'
# end_time = '2017-05-03 12:00:00'
#
# aor = channel_detection[channel_detection['signal_status'] == 'red'].reset_index(drop=True)
# aor = aor.groupby(by=['signal_id', 'event_parameter', 'cycle_num'], as_index=False).agg({"occupancy": ['mean', 'std', 'count'], "headway": ['mean', 'std']})
# aor.columns = ['signal_id', 'event_parameter', 'cycle_num', 'avg_occupancy', 'std_occupancy', 'count', 'avg_headway', 'std_headway']
# aor = aor.join(phase_signal[['cycle_num', 'cycle_len', 'timestamp']].set_index('cycle_num'), on='cycle_num')
# aor = aor[(aor['timestamp'].dt.hour >= 9) & (aor['timestamp'].dt.hour <= 11)].reset_index(drop=True)
#
# max_aor = aor[['signal_id', 'event_parameter', 'count']].groupby(by=['signal_id', 'event_parameter'], as_index=False).quantile(.95)
#
# aor_count.append(max_aor)
#
# aor['count'].describe()
# test = aor.sort_values('count', ascending=False)
# detection, signal = load_detection_signal_log(1505)
# test_34 = detection[detection['event_parameter'] == 34].reset_index(drop=True)
# load the intersection sequence
# intersection_seq = pd.read_csv('./GIS/SR_436_intersection_sequence.csv', index_col=0)
# signal_list = list(intersection_seq.Intersec_1.str[5:9])
# len(signal_list)
# signal_id = 1815

# load ATSPM detector configuration and ATSPM intersection
detector_config = pd.read_excel(r'J:/ATSPM_CNN_LSTM/Data/Detector_Checklist_join.xlsx', sheet_name='All')
# atspm_intersections = pd.read_excel(r'J:/ATSPM_CNN_LSTM/Data/Signal Location-Revise.xlsx', sheet_name='Queue')


# extract the list of intersections based on what raw data we have in the folder
path = os.getcwd()
data_path = os.path.join(path, 'Data\Raw_Data\Test')
files = os.listdir(data_path) # Get files path within the folder
intersections = [int(f[4:8]) for f in files if f[0:3] == 'Raw']


if __name__ == '__main__':
    detector_config_r = clean_detector_config(detector_config, intersections)

    # call the function to calculate the max_aor to verify the detector location
    total_aor_count = channel_volume_max_aor(intersections, detector_config_r, '2019-09-04 07:00:00', '2019-09-04 21:00:00')

    # join the max_aor to the detector configuration
    detector_config_r = pd.merge(detector_config_r, total_aor_count, how='left', left_on=['SignalID', 'Det_Channel'], right_on=['signal_id', 'event_parameter'])
    detector_config_r.rename(columns={'count': 'max_aor'}, inplace=True)

    # configure the attribute of "Stop_bar"
    detector_config_r.loc[:, 'Stop_bar'] = 0
    detector_config_r.loc[(detector_config_r['max_aor'] <= 3) | (detector_config_r['Delay'] != 999) |
                          (detector_config_r['Switch'] != 999) | (detector_config_r['Volume_Movement'].str.contains('|'.join(['L', 'R']))), 'Stop_bar'] = 1

    # save to csv file
    detector_config_r.loc[detector_config_r['Stop_bar'] == 0, 'modified_distance'] = detector_config_r.loc[detector_config_r['Stop_bar'] == 0, 'max_aor'] * 25
    detector_config_r.loc[detector_config_r['Stop_bar'] == 1, 'modified_distance'] = 0
    detector_config_r.to_csv('./Data/Detector_Checklist_final_test.csv', sep=',')

    # detector_config = pd.merge(detector_config, detector_config_r, how='left', left_on=['SignalID', 'Det_Channel'], right_on=['SignalID', 'Det_Channel'])
    # detector_config.to_csv('./Data/test.csv', sep=',')

