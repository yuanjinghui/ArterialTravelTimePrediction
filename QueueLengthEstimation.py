"""
Created on Fri Jan 17 20:01:56 2020

@author: ji758507
"""
import pandas as pd
import math


def load_detection_signal_log(signal_id):
    """
    Load vehicle detection and signal log data for the given intersection id.
    :param signal_id: signal id
    :return: two dataframes include vehicle actuation and signal log information, respeectively.
    """
    raw = pd.read_csv('./Data/Raw_Data/SR436/Raw_{}.csv'.format(signal_id), names=['signal_id', 'timestamp', 'event_code', 'event_parameter'])
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    raw = raw.sort_values(by=['timestamp'])
    raw = raw[(raw['timestamp'] >= '2018-07-01 00:00:00') & (raw['timestamp'] <= '2019-01-01 00:00:00')].reset_index(drop=True)
    detection = raw[raw['event_code'].isin([81, 82])]
    detection = detection.sort_values(by=['event_parameter', 'timestamp', 'event_code'])
    # signal = raw[raw['event_code'].isin(list(range(1, 12)))]
    signal = raw[raw['event_code'].isin([1, 8, 10])]
    signal = signal.sort_values(by=['event_parameter', 'timestamp', 'event_code'])

    return detection, signal


def gen_signal_cycle_info(signal, phase, movement):
    """
    Generate signal cycle information based on the intersection signal timing data and given phase number
    :param signal: signal timing data for given intersection
    :param phase: given phase number
    :param movement: given traffic movement
    :return: signal cycle information
    """
    # extract signal timing data for selected phase id
    phase_signal = signal[signal['event_parameter'] == phase].reset_index(drop=True)

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
    # lag   lag   lag2   lag3   lag4
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
        = phase_signal.loc[
        (phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'lag2_timestamp']
    phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'EOG'] \
        = phase_signal.loc[
        (phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'lag3_timestamp']
    phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'BOY'] \
        = phase_signal.loc[
        (phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'lag3_timestamp']
    phase_signal.loc[(phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'EOY'] \
        = phase_signal.loc[
        (phase_signal['lag_event_code'] == 8) & (phase_signal['lag2_event_code'] == 1) & (phase_signal['lag3_event_code'] == 8) & (phase_signal['lag4_event_code'] == 10), 'lag4_timestamp']

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
    phase_signal['movement'] = movement

    return phase_signal


def gen_channel_detection(phase_signal, detection, channel_list, location_pattern, advance_detector_loc):
    """
    Generate the major variables based on the detection and signal timing data
    :param phase_signal: signal timing data for the given specific signal phase
    :param detection: detection data for the given intersection
    :param channel_list: list of detector channels for data generation
    :param location_pattern: location pattern
    :param advance_detector_loc: location of the advanced detectors
    :return: generated data table
    """
    # prepare the base cycle signal timing data
    var_data = phase_signal[['signal_id', 'movement', 'cycle_num', 'cycle_len', 'BOC', 'EOC', 'BOR', 'EOR', 'BOG', 'EOG', 'green_time', 'yellow_time', 'red_time']].copy()
    for j in channel_list:
        # channel = channel_list[j]
        channel_detection = detection[detection['event_parameter'] == j].reset_index(drop=True)
        # stopbar_advance = detector_table.loc[detector_table['Det_Channel'] == j, 'Stop_bar'].astype(int).tolist()[0]

        # delete error data (event_code = lead_event_code = 81 or event_code = lag_event_code = 82)
        channel_detection['lag_event_code'] = channel_detection['event_code'].shift(-1)
        channel_detection['lead_event_code'] = channel_detection['event_code'].shift(1)
        channel_detection = channel_detection[~(((channel_detection['event_code'] == 81) & (channel_detection['lead_event_code'] == 81)) | (
                (channel_detection['event_code'] == 82) & (channel_detection['lag_event_code'] == 82)))][
            ['signal_id', 'timestamp', 'event_code', 'event_parameter', 'lag_event_code', 'lead_event_code']].reset_index(drop=True)

        channel_detection['lag_timestamp'] = channel_detection['timestamp'].shift(-1)
        channel_detection['lead_timestamp'] = channel_detection['timestamp'].shift(1)

        # only keep the detector on rows
        channel_detection = channel_detection[
            (channel_detection['event_code'] == 82) & (channel_detection['lag_event_code'].notnull()) & (channel_detection['lead_event_code'].notnull())].reset_index(drop=True)

        channel_detection['occupancy'] = (channel_detection['lag_timestamp'] - channel_detection['timestamp']).dt.total_seconds()
        channel_detection['gap'] = (channel_detection['timestamp'] - channel_detection['lag_timestamp'].shift(1)).dt.total_seconds()
        channel_detection['headway'] = channel_detection['occupancy'] + channel_detection['gap']
        channel_detection['lag_gap'] = channel_detection['gap'].shift(-1)

        # filter out error data
        channel_detection = channel_detection[(channel_detection['occupancy'] > 0) & (channel_detection['gap'] > 0) & (channel_detection['occupancy'] < 3600) & (channel_detection['lag_gap'].notnull())].reset_index(drop=True)

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

        # groupby cycle number to get cycle volume for every cycle
        channel_vol_cycle = channel_detection.groupby(by=['cycle_num'], as_index=False).size().reset_index(name='cycle_volume_{}_{}'.format(location_pattern, j))

        channel_aog = channel_detection.groupby(by=['green_cycle_num'], as_index=False).agg({"occupancy": ['mean', 'std'], "headway": ['mean', 'std', 'count']})
        channel_aog.columns = ['cycle_num', 'avg_occupancy_green_{}_{}'.format(location_pattern, j),
                               'std_occupancy_green_{}_{}'.format(location_pattern, j),
                               'avg_headway_green_{}_{}'.format(location_pattern, j),
                               'std_headway_green_{}_{}'.format(location_pattern, j),
                               'aog_{}_{}'.format(location_pattern, j)]

        channel_aor = channel_detection.groupby(by=['red_cycle_num'], as_index=False).agg({"occupancy": ['mean', 'std'], "headway": ['mean', 'std', 'count']})
        channel_aor.columns = ['cycle_num', 'avg_occupancy_red_{}_{}'.format(location_pattern, j),
                               'std_occupancy_red_{}_{}'.format(location_pattern, j),
                               'avg_headway_red_{}_{}'.format(location_pattern, j),
                               'std_headway_red_{}_{}'.format(location_pattern, j),
                               'aor_{}_{}'.format(location_pattern, j)]

        channel_aoy = channel_detection.groupby(by=['yellow_cycle_num'], as_index=False).agg({"occupancy": ['mean', 'count']})
        channel_aoy.columns = ['cycle_num', 'avg_occupancy_yellow_{}_{}'.format(location_pattern, j),
                               'aoy_{}_{}'.format(location_pattern, j)]

        # identify the A, B, C point
        # A: if the advance detector shows high occupancy in the last detection during red, then A exist. Otherwise, A doesnot exist. Using input-output method (30*AOR).
        # B: if A exist, B exist. Time B equals to the first detection during green.
        # C: the time of the first detection with high headway during green
        if location_pattern in (2, 3):
            # calculate the occupancy of the last detection during red
            identify_A = channel_detection[['red_cycle_num', 'occupancy', 'timestamp', 'lag_timestamp']].groupby(by=['red_cycle_num'], as_index=False).last()

            # calculate the mean of detections during red except the last detection
            identify_A_mean = channel_detection.loc[channel_detection['red_cycle_num'].notnull(), ['red_cycle_num', 'occupancy']].groupby(by=['red_cycle_num'], as_index=False).apply(lambda x: x[0:-1].mean())
            identify_A_mean = identify_A_mean[identify_A_mean['red_cycle_num'].notnull()]
            identify_A_mean.rename(columns={'occupancy': 'avg_occupancy'}, inplace=True)
            identify_A = pd.merge(identify_A, identify_A_mean, how='left', left_on=['red_cycle_num'], right_on=['red_cycle_num'])

            # calculate the difference between the last detection and the mean value of previous detection during red
            identify_A['diff_occupancy'] = identify_A['occupancy'] - identify_A['avg_occupancy']

            # if the difference is greater than 5, then we assume the A exist
            identify_A['A_exist'] = identify_A['diff_occupancy'].apply(lambda x: 1 if x >= 3 else 0)
            identify_A.loc[identify_A['A_exist'] == 1, 'A_time'] = identify_A.loc[identify_A['A_exist'] == 1, 'timestamp']
            identify_A.loc[identify_A['A_exist'] == 1, 'B_time'] = identify_A.loc[identify_A['A_exist'] == 1, 'lag_timestamp']

            identify_A = identify_A[['red_cycle_num', 'A_exist', 'A_time', 'B_time']]
            identify_A.rename(columns={'red_cycle_num': 'cycle_num', 'A_exist': 'A_exist_{}_{}'.format(location_pattern, j), 'A_time': 'A_time_{}_{}'.format(location_pattern, j), 'B_time': 'B_time_{}_{}'.format(location_pattern, j)}, inplace=True)

            # when A exist, check the detections (from the second to the last) during green, find out the first detection with the lag_gap greater than 2.5 second.
            identify_C = channel_detection.join(identify_A[['cycle_num', 'A_exist_{}_{}'.format(location_pattern, j)]].set_index('cycle_num'), on='green_cycle_num')

            if len(identify_C[identify_C['A_exist_{}_{}'.format(location_pattern, j)] == 1]) > 0:

                # channel_aor will be recalculated based on two parts (consider the last red detection may have extremely high occupancy)
                A_list = identify_A[identify_A['A_exist_{}_{}'.format(location_pattern, j)] == 1]['cycle_num'].tolist()
                channel_aor_1 = channel_detection[channel_detection['cycle_num'].isin(A_list)].groupby(by=['red_cycle_num'], as_index=False).apply(lambda group: group.iloc[:-1, :])
                channel_aor_1 = channel_aor_1.groupby(by=['red_cycle_num'], as_index=False).agg({"occupancy": ['mean', 'std'], "headway": ['mean', 'std']})
                channel_aor_1.columns = ['cycle_num', 'avg_occupancy_red_{}_{}'.format(location_pattern, j), 'std_occupancy_red_{}_{}'.format(location_pattern, j), 'avg_headway_red_{}_{}'.format(location_pattern, j), 'std_headway_red_{}_{}'.format(location_pattern, j)]

                channel_aor_2 = channel_detection[~channel_detection['cycle_num'].isin(A_list)].groupby(by=['red_cycle_num'], as_index=False).agg({"occupancy": ['mean', 'std'], "headway": ['mean', 'std']})
                channel_aor_2.columns = ['cycle_num', 'avg_occupancy_red_{}_{}'.format(location_pattern, j), 'std_occupancy_red_{}_{}'.format(location_pattern, j), 'avg_headway_red_{}_{}'.format(location_pattern, j), 'std_headway_red_{}_{}'.format(location_pattern, j)]
                channel_aor_tem = pd.concat([channel_aor_1, channel_aor_2], ignore_index=True).sort_values(by=['cycle_num'])
                channel_aor = channel_aor_tem.join(channel_aor[['cycle_num', 'aor_{}_{}'.format(location_pattern, j)]].set_index('cycle_num'), on='cycle_num')

                # identify C point
                identify_C = identify_C[identify_C['A_exist_{}_{}'.format(location_pattern, j)] == 1].reset_index(drop=True)
                identify_C_backup = identify_C[['cycle_num', 'lag_gap', 'timestamp']].groupby(by=['cycle_num'], as_index=False).last()
                identify_C_backup.rename(columns={'timestamp': 'C_time_backup_{}_{}'.format(location_pattern, j)}, inplace=True)

                # identify_C = identify_C[['cycle_num', 'lag_gap', 'timestamp']].groupby(by=['cycle_num'], as_index=False).apply(lambda group: group.iloc[1:, :])
                identify_C = identify_C[['cycle_num', 'lag_gap', 'timestamp']].groupby(by=['cycle_num'], as_index=False).apply(lambda group: group.loc[group['lag_gap'] > 2.5, :])

                # there is C exist
                if len(identify_C) > 0:
                    identify_C = identify_C[['cycle_num', 'lag_gap', 'timestamp']].groupby(by=['cycle_num'], as_index=False).first()
                    identify_C.rename(columns={'timestamp': 'C_time_{}_{}'.format(location_pattern, j)}, inplace=True)

                    identify_A = pd.merge(identify_A, identify_C[['cycle_num', 'C_time_{}_{}'.format(location_pattern, j)]], how='left', left_on=['cycle_num'], right_on=['cycle_num'])
                    identify_A = pd.merge(identify_A, identify_C_backup[['cycle_num', 'C_time_backup_{}_{}'.format(location_pattern, j)]], how='left', left_on=['cycle_num'], right_on=['cycle_num'])

                    # if cannot find C point, then take the last green detection as C point
                    identify_A.loc[(identify_A['A_exist_{}_{}'.format(location_pattern, j)] == 1) & (identify_A['C_time_{}_{}'.format(location_pattern, j)].isnull()), 'C_time_{}_{}'.format(location_pattern, j)] = identify_A['C_time_backup_{}_{}'.format(location_pattern, j)]
                    identify_A = identify_A.drop(['C_time_backup_{}_{}'.format(location_pattern, j)], axis=1)

                # there is no C point exist, use C backup instead
                else:
                    identify_A = pd.merge(identify_A, identify_C_backup[['cycle_num', 'C_time_backup_{}_{}'.format(location_pattern, j)]], how='left', left_on=['cycle_num'], right_on=['cycle_num'])
                    # if cannot find C point, then take the last green detection as C point
                    identify_A.rename(columns={'C_time_backup_{}_{}'.format(location_pattern, j): 'C_time_{}_{}'.format(location_pattern, j)}, inplace=True)

                # Calculate the theoretical maximum queue length (AOR + AOG (Before Time C))
                theory_Max_queue = pd.merge(channel_detection, identify_A[['cycle_num', 'A_exist_{}_{}'.format(location_pattern, j), 'C_time_{}_{}'.format(location_pattern, j)]], how='left', left_on=['cycle_num'], right_on=['cycle_num'])
                theory_Max_queue = theory_Max_queue.loc[theory_Max_queue['A_exist_{}_{}'.format(location_pattern, j)] == 1, ]
                theory_Max_queue = theory_Max_queue[['cycle_num', 'timestamp', 'C_time_{}_{}'.format(location_pattern, j)]].groupby(by=['cycle_num'], as_index=False).apply(lambda group: group.loc[group['timestamp'] <= group['C_time_{}_{}'.format(location_pattern, j)], :])
                theory_Max_queue = theory_Max_queue.groupby(by=['cycle_num'], as_index=False).size().reset_index(name='Theory_Max_queue_{}_{}'.format(location_pattern, j))

                # combine all the variables for every cycle
                dfs = [var_data, channel_vol_cycle, channel_aog, channel_aor, channel_aoy, identify_A, theory_Max_queue]
                dfs = [df.set_index('cycle_num') for df in dfs]
                var_data = dfs[0].join(dfs[1:])
                var_data['cycle_num'] = var_data.index

                var_data.loc[var_data['Theory_Max_queue_{}_{}'.format(location_pattern, j)].notnull(), 'Theory_Max_queue_{}_{}'.format(location_pattern, j)] = var_data['Theory_Max_queue_{}_{}'.format(location_pattern, j)] * 25
                var_data.loc[var_data['Theory_Max_queue_{}_{}'.format(location_pattern, j)].isnull(), 'Theory_Max_queue_{}_{}'.format(location_pattern, j)] = var_data['aor_{}_{}'.format(location_pattern, j)] * 25

                # calculate maximum queue length and queuing shockwave speed (v1) based on shockwave theory
                # assume jam density (Kj) is 13 vehicles per 330 ft, which equals to 211.2 vehicles per mile per lane
                # assume effective vehicle length equals to 25 ft
                # queuing shockwave speed (v1) = (0-Qa)/(Kj-Ka), refer to 'Real-time queue length estimation for congested signalized intersections'
                var_data['Qa_{}_{}'.format(location_pattern, j)] = (1 / var_data['avg_headway_red_{}_{}'.format(location_pattern, j)]) * 3600
                var_data['red_arrival_speed_{}_{}'.format(location_pattern, j)] = (25 / var_data['avg_occupancy_red_{}_{}'.format(location_pattern, j)]) * 0.681818  # convert ft/s to mile/hour
                var_data['Ka_{}_{}'.format(location_pattern, j)] = var_data['Qa_{}_{}'.format(location_pattern, j)] / var_data['red_arrival_speed_{}_{}'.format(location_pattern, j)]
                var_data['queuing_shockwave_spd_{}_{}'.format(location_pattern, j)] = ((0 - var_data['Qa_{}_{}'.format(location_pattern, j)]) / (211.2 - var_data['Ka_{}_{}'.format(location_pattern, j)])) * 1.46667  # convert mile/hour to ft/s

                # assume saturation flow rate Qm, and saturation density Km are known as 1900 vehicles/hour/lane and speed limit equals to 45 mph, density equals to 42 vehicles per mile per lane
                # discharge_shockwave_spd = (1900-0)/(42-192)
                var_data['discharge_shockwave_spd_{}_{}'.format(location_pattern, j)] = advance_detector_loc / (var_data['B_time_{}_{}'.format(location_pattern, j)] - var_data['BOG']).dt.total_seconds()
                var_data['departure_shockwave_spd_{}_{}'.format(location_pattern, j)] = ((1900 - var_data['Qa_{}_{}'.format(location_pattern, j)]) / (42 - var_data['Ka_{}_{}'.format(location_pattern, j)])) * 1.46667 # convert mile/hour to ft/s

                var_data.loc[var_data['A_exist_{}_{}'.format(location_pattern, j)] == 1, 'max_queue_length_{}_{}'.format(location_pattern, j)] = \
                        advance_detector_loc + (var_data['C_time_{}_{}'.format(location_pattern, j)] - var_data['B_time_{}_{}'.format(location_pattern, j)]).dt.total_seconds() / ((1 / abs(var_data['discharge_shockwave_spd_{}_{}'.format(location_pattern, j)])) + (1 / abs(var_data['departure_shockwave_spd_{}_{}'.format(location_pattern, j)])))
                var_data.loc[var_data['A_exist_{}_{}'.format(location_pattern, j)] != 1, 'max_queue_length_{}_{}'.format(location_pattern, j)] = var_data['aor_{}_{}'.format(location_pattern, j)] * 25

                # verity the maximum queue length is smaller than the theory max queue
                var_data.loc[var_data['max_queue_length_{}_{}'.format(location_pattern, j)] > var_data['Theory_Max_queue_{}_{}'.format(location_pattern, j)], 'max_queue_length_{}_{}'.format(location_pattern, j)] = var_data['Theory_Max_queue_{}_{}'.format(location_pattern, j)]

                # fill null
                var_data.loc[:, ['cycle_volume_{}_{}'.format(location_pattern, j), 'aog_{}_{}'.format(location_pattern, j), 'aor_{}_{}'.format(location_pattern, j), 'aoy_{}_{}'.format(location_pattern, j), 'max_queue_length_{}_{}'.format(location_pattern, j)]] = \
                        var_data[['cycle_volume_{}_{}'.format(location_pattern, j), 'aog_{}_{}'.format(location_pattern, j), 'aor_{}_{}'.format(location_pattern, j), 'aoy_{}_{}'.format(location_pattern, j), 'max_queue_length_{}_{}'.format(location_pattern, j)]].fillna(0)

            # if A point does not exist
            else:
                dfs = [var_data, channel_vol_cycle, channel_aog, channel_aor, channel_aoy]
                dfs = [df.set_index('cycle_num') for df in dfs]
                var_data = dfs[0].join(dfs[1:])
                var_data['cycle_num'] = var_data.index

                var_data['Qa_{}_{}'.format(location_pattern, j)] = (1 / var_data['avg_headway_red_{}_{}'.format(location_pattern, j)]) * 3600
                var_data['red_arrival_speed_{}_{}'.format(location_pattern, j)] = (25 / var_data['avg_occupancy_red_{}_{}'.format(location_pattern, j)]) * 0.681818  # convert ft/s to mile/hour
                var_data['Ka_{}_{}'.format(location_pattern, j)] = var_data['Qa_{}_{}'.format(location_pattern, j)] / var_data['red_arrival_speed_{}_{}'.format(location_pattern, j)]
                var_data['queuing_shockwave_spd_{}_{}'.format(location_pattern, j)] = \
                    ((0 - var_data['Qa_{}_{}'.format(location_pattern, j)]) / (211.2 - var_data['Ka_{}_{}'.format(location_pattern, j)])) * 1.46667

                var_data['max_queue_length_{}_{}'.format(location_pattern, j)] = var_data['aor_{}_{}'.format(location_pattern, j)] * 25
                var_data.loc[:, ['cycle_volume_{}_{}'.format(location_pattern, j), 'aog_{}_{}'.format(location_pattern, j), 'aor_{}_{}'.format(location_pattern, j), 'aoy_{}_{}'.format(location_pattern, j), 'max_queue_length_{}_{}'.format(location_pattern, j)]] = \
                    var_data[['cycle_volume_{}_{}'.format(location_pattern, j), 'aog_{}_{}'.format(location_pattern, j), 'aor_{}_{}'.format(location_pattern, j), 'aoy_{}_{}'.format(location_pattern, j),
                     'max_queue_length_{}_{}'.format(location_pattern, j)]].fillna(0)


        else:
            dfs = [var_data, channel_vol_cycle, channel_aog, channel_aor, channel_aoy]
            dfs = [df.set_index('cycle_num') for df in dfs]
            var_data = dfs[0].join(dfs[1:])
            var_data['cycle_num'] = var_data.index

            var_data.loc[:, ['cycle_volume_{}_{}'.format(location_pattern, j), 'aog_{}_{}'.format(location_pattern, j), 'aor_{}_{}'.format(location_pattern, j), 'aoy_{}_{}'.format(location_pattern, j)]] = \
                var_data[['cycle_volume_{}_{}'.format(location_pattern, j), 'aog_{}_{}'.format(location_pattern, j), 'aor_{}_{}'.format(location_pattern, j), 'aoy_{}_{}'.format(location_pattern, j)]].fillna(0)

        print(j)

    return var_data


def gen_through_var(through_detector_table, detection, phase_signal):

    min_through_loc_pattern = min(through_detector_table.loc[through_detector_table['location_pattern'] > 1, 'location_pattern'].unique().astype(int).tolist())
    advance_through_detector_table = through_detector_table[through_detector_table['location_pattern'] == min_through_loc_pattern].reset_index(drop=True)
    advance_detector_loc = int(math.ceil(advance_through_detector_table.modified_distance.mean() / 5.0) * 5.0)
    advance_through_channel_list = advance_through_detector_table['Det_Channel'].unique().tolist()

    # prepare the raw variables
    var_data = gen_channel_detection(phase_signal, detection, advance_through_channel_list, min_through_loc_pattern, advance_detector_loc)
    var_data['cycle_volume'] = var_data.iloc[:, var_data.columns.str.contains('cycle_volume')].sum(axis=1)

    # aggregate the pog, poy, por
    var_data['green_ratio'] = var_data['green_time'] / var_data['cycle_len']

    var_data['aog'] = var_data.iloc[:, var_data.columns.str.contains('aog')].sum(axis=1)
    var_data['aoy'] = var_data.iloc[:, var_data.columns.str.contains('aoy')].sum(axis=1)
    var_data['aor'] = var_data.iloc[:, var_data.columns.str.contains('aor')].sum(axis=1)

    var_data['pog'] = var_data['aog'] / var_data['cycle_volume']
    var_data['poy'] = var_data['aoy'] / var_data['cycle_volume']
    var_data['por'] = var_data['aor'] / var_data['cycle_volume']

    var_data['aogr'] = var_data['pog'] / var_data['green_time']
    var_data['aoyr'] = var_data['poy'] / var_data['yellow_time']
    var_data['aorr'] = var_data['por'] / var_data['red_time']

    var_data['platoon_ratio'] = var_data['aogr'] * var_data['cycle_len']

    # aggregate occupancy and headway related variables
    var_data['avg_occupancy_green'] = var_data.iloc[:, var_data.columns.str.contains('avg_occupancy_green')].mean(axis=1)
    var_data['std_occupancy_green'] = var_data.iloc[:, var_data.columns.str.contains('std_occupancy_green')].mean(axis=1)
    var_data['avg_headway_green'] = var_data.iloc[:, var_data.columns.str.contains('avg_headway_green')].mean(axis=1)
    var_data['std_headway_green'] = var_data.iloc[:, var_data.columns.str.contains('std_headway_green')].mean(axis=1)

    var_data['avg_occupancy_red'] = var_data.iloc[:, var_data.columns.str.contains('avg_occupancy_red')].mean(axis=1)
    var_data['std_occupancy_red'] = var_data.iloc[:, var_data.columns.str.contains('std_occupancy_red')].mean(axis=1)
    var_data['avg_headway_red'] = var_data.iloc[:, var_data.columns.str.contains('avg_headway_red')].mean(axis=1)
    var_data['std_headway_red'] = var_data.iloc[:, var_data.columns.str.contains('std_headway_red')].mean(axis=1)

    # aggregate shockwave characteristics
    var_data['A_exist'] = var_data.iloc[:, var_data.columns.str.contains('A_exist')].max(axis=1)
    var_data['max_queue_length'] = var_data.iloc[:, var_data.columns.str.contains('max_queue_length')].max(axis=1)
    var_data['shock_wave_area'] = (var_data['max_queue_length'] * var_data['red_time']) / (2 * 5280)  # unit is mile.s
    # convert the unit of max_queue_length from ft to veh
    var_data['max_queue_length'] = var_data['max_queue_length'] / 25  # unit is veh
    var_data['queuing_shockwave_spd'] = var_data.iloc[:, var_data.columns.str.contains('queuing_shockwave_spd')].mean(axis=1)

    # exclude all the detector specific variables
    var_data = var_data.loc[:, ~var_data.columns.str.contains('|'.join(['_1_', '_2_', '_3_']))]

    return var_data


def gen_approach_list(atspm_detector_config):
    """

    :param atspm_detector_config:
    :return:
    """
    atspm_detector_config = atspm_detector_config[atspm_detector_config['Volume'].isin([1, 2, 3])].reset_index(drop=True)

    # create a column indicates the detector location pattern
    # stop_bar  volume  location_pattern
    # 1         1       1
    # 0         1       2
    # 0         2       3
    # 0         3       4
    atspm_detector_config.loc[atspm_detector_config['Stop_bar'] == 1, 'location_pattern'] = 1
    atspm_detector_config.loc[(atspm_detector_config['Stop_bar'] == 0) & (atspm_detector_config['Volume'] == 1), 'location_pattern'] = 2
    atspm_detector_config.loc[(atspm_detector_config['Stop_bar'] == 0) & (atspm_detector_config['Volume'] == 2), 'location_pattern'] = 3
    atspm_detector_config.loc[(atspm_detector_config['Stop_bar'] == 0) & (atspm_detector_config['Volume'] == 3), 'location_pattern'] = 4
    return atspm_detector_config


# # check the configuration of detector locations for the major approaches
# test = atspm_detector_config.groupby(by=['SignalID', 'Movement'], as_index=False).agg({"location_pattern": 'mean', "Volume": 'mean'})
# test1 = intersection_approaches_dict.groupby(by=['signal_id', 'Mj_Dir', 'ES_seq'], as_index=False).size().reset_index(name='count')
# test = test.join(test1.set_index('signal_id'), on='SignalID')
# # test = test[(test['Movement'].str.contains('T')) & (test.apply(lambda x: x.Movement[0] in x.Mj_Dir, axis=1))].reset_index(drop=True)
# # test = test[(test['Movement'].str.contains('L')) & (test.apply(lambda x: x.Movement[0] in x.Mj_Dir, axis=1))].reset_index(drop=True)
#
# # check how many approaches are operated with left turn permissive control
# test = test[test.apply(lambda x: x.Movement[0] in x.Mj_Dir, axis=1)].reset_index(drop=True)
# test['direction'] = test['Movement'].str[0]
# test2 = test.groupby(by=['SignalID', 'direction'], as_index=False).size().reset_index(name='count')
# test3 = intersection_approaches_dict[['signal_id', 'direction', 'Seg_ID']].copy()
# test3['direction_r'] = test3['direction'].str[0]
# test2 = pd.merge(test2, test3, how='left', left_on=['SignalID', 'direction'], right_on=['signal_id', 'direction_r'])
# test2 = test2[test2['signal_id'].notnull()].reset_index(drop=True)
# intersection_approaches_dict = intersection_approaches_dict.join(test2[['Seg_ID', 'count']].set_index('Seg_ID'), on='Seg_ID')
# intersection_approaches_dict.to_csv('./Data/intersection_approaches_dict.csv', sep=',')


# cycle definition: red + green + yellow (10, 1, 8)
def generate_data_for_approaches(atspm_detector_config):
    """

    :param intersection_approaches_dict:
    :param atspm_detector_config:
    :param weather_path:
    :return:
    """
    raw_data = []
    intersection_approaches_dict = atspm_detector_config.loc[atspm_detector_config['Direction'] != '999', ['SignalID', 'Direction']].drop_duplicates().reset_index(drop=True)

    for i in range(len(intersection_approaches_dict)):
        signal_id = intersection_approaches_dict.loc[i, 'SignalID']
        direction = intersection_approaches_dict.loc[i, 'Direction'][0:1]
        detector_table = atspm_detector_config.loc[(atspm_detector_config['SignalID'] == signal_id) & (atspm_detector_config['Movement'].str.contains(direction))].reset_index(drop=True)
        phase_list = detector_table['Phase'].unique().tolist()

        detection, signal = load_detection_signal_log(signal_id)

        if len(phase_list) == 2:
            # --------------------------------------------------------------------------------------------------------------------------------------------
            # process data for the through phase
            # --------------------------------------------------------------------------------------------------------------------------------------------
            through_detector_table = atspm_detector_config.loc[(atspm_detector_config['SignalID'] == signal_id) & (atspm_detector_config['Movement'] == direction + 'T')].reset_index(drop=True)
            through_phase = through_detector_table['Phase'].unique().tolist()[0]

            phase_signal = gen_signal_cycle_info(signal, through_phase, direction + 'T')

            var_data = gen_through_var(through_detector_table, detection, phase_signal)

        # if this approach is operating with permissive left turn signal
        if len(phase_list) == 1:
            # if Volume_Movement does not include direction + 'L', which means that this approach has no left turning lanes
            # if Volume_Movement include direction + 'L', which means that this approach has left turning lanes with permissive control
            lane_movement_list = detector_table['Volume_Movement'].unique().tolist()
            combined = '_'.join(lane_movement_list)

            # permissive left turn
            if 'L' in combined:
                # --------------------------------------------------------------------------------------------------------------------------------------------
                # process data for the through phase
                # --------------------------------------------------------------------------------------------------------------------------------------------
                through_detector_table = atspm_detector_config.loc[(atspm_detector_config['SignalID'] == signal_id) & (atspm_detector_config['Volume_Movement'] == direction + 'T')].reset_index(drop=True)
                through_phase = through_detector_table['Phase'].unique().tolist()[0]

                phase_signal = gen_signal_cycle_info(signal, through_phase, direction + 'T')

                var_data = gen_through_var(through_detector_table, detection, phase_signal)

            # no left turning lanes
            else:
                # --------------------------------------------------------------------------------------------------------------------------------------------
                # process data for the through phase
                # --------------------------------------------------------------------------------------------------------------------------------------------
                through_detector_table = atspm_detector_config.loc[(atspm_detector_config['SignalID'] == signal_id) & (atspm_detector_config['Movement'] == direction + 'T')].reset_index(drop=True)
                through_phase = through_detector_table['Phase'].unique().tolist()[0]

                phase_signal = gen_signal_cycle_info(signal, through_phase, direction + 'T')

                var_data = gen_through_var(through_detector_table, detection, phase_signal)

        raw_data.append(var_data)
        # raw_data = raw_data.append(var_data)
        # raw_data = pd.concat([raw_data, var_data], ignore_index=True)
        print("approach id " + str(i))

    data = pd.concat(raw_data, ignore_index=True)

    return data


# raw_data = generate_data_for_approaches(intersection_approaches_dict, atspm_detector_config)
# raw_data = pd.read_csv('./Data/raw_data.csv', index_col=0)
# raw_data.iloc[:, 4:10] = raw_data.iloc[:, 4:10].apply(pd.to_datetime, errors='coerce')
# atspm_detector_config = pd.read_csv('./Data/Detector_Checklist_final.csv', index_col=0)
atspm_detector_config_SR436_Selected = pd.read_csv('./Data/Detector_Checklist_final_SR436_Selected.csv', index_col=0)
# time_steps = 6
# time_steps = 26
# weather_path = './Data/Weather Data/1619773.csv'
# i = 4
# j = 2
# location_pattern = 2

if __name__ == '__main__':

    atspm_detector_config = gen_approach_list(atspm_detector_config_SR436_Selected)
    raw_data = generate_data_for_approaches(atspm_detector_config)
    # raw_data.to_csv('./Data/raw_data.csv', sep=',')
    raw_data.to_csv('./Data/raw_data_SR436_Selected.csv', sep=',')
    # intersection_approaches_dict.to_csv('./Data/intersection_approaches_dict.csv', sep=',')

    # weather_data = read_weather_data(weather_path, 5)
    #
    # convert_cycle_to_minutes(raw_data, intersection_approaches_dict, time_steps, weather_data)

    # intersection_entr, segment = gen_y_label_data(intersection_approaches_dict, intersection_entrance_crashes, segments_crashes)
    #
    # intersection_entr.to_csv('./Data/processed_intersection_crash.csv', sep=',')
    # segment.to_csv('./Data/processed_segment_crash.csv', sep=',')
