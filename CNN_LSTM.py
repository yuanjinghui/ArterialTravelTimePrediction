"""
Created on Mon Jul 21 20:01:56 2019

@author: ji758507
"""
import h5py
import keras
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Dense, Dropout, CuDNNLSTM
from keras.callbacks import Callback, EarlyStopping
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
import numpy as np
from keras.optimizers import Adam, Nadam, SGD, RMSprop, Adadelta
import talos as ta
from keras.callbacks import TensorBoard
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use id from $ nvidia-smi


def load_data_for_convLSTM(data_file):
    # two variables
    # with h5py.File('Data/modeling_data_15.h5', 'a') as f:
    # with h5py.File('Data/modeling_data_10.h5', 'r') as f:
    with h5py.File(data_file, 'r') as f:
    # all variables
    # with h5py.File('Data/modeling_data.h5', 'r') as f:
        print(list(f.keys()))
        x_train = f['x_train'][:]
        y_train = f['y_train'][:]
        x_test = f['x_test'][:]
        y_test = f['y_test'][:]
        f.close()
    return x_train, y_train, x_test, y_test


# x_train.dtype
# y_test.dtype
def label_reshape(y_pred):
    y_pred = y_pred.reshape(y_pred.shape[0], 1, -1)
    y_pred_r = np.moveaxis(y_pred, -1, 1)
    y_pred_r = y_pred_r.reshape(-1, 1)
    return y_pred_r


def x_reshape(x_train):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], -1)
    a = x_train.shape[1]
    b = x_train.shape[2]
    x_train = np.moveaxis(x_train, -1, 1)
    x_train = x_train.reshape(-1, a, b)
    return x_train


tensorboard = TensorBoard(log_dir='./logs',
                 histogram_freq=0, batch_size=5000,
                 write_graph=False,
                 write_images=False)


def get_optimizer(name):
    if name == 'SGD':
        return SGD()
    if name == 'RMSprop':
        return RMSprop()
    if name == 'Adadelta':
        return Adadelta()
    if name == 'Adam':
        return Adam()
    if name == 'Nadam':
        return Nadam()


# first we have to make sure to input data and params into the function
def corridor_cnn_lstm(x_train, y_train, x_val, y_val, params):
    print(x_train.shape)
    n_features = x_train.shape[2]
    # split 30 time steps into 5*6
    n_steps, n_length = 5, 6
    x_train = x_train.reshape((x_train.shape[0], n_steps, n_length, n_features))
    x_val = x_val.reshape((x_val.shape[0], n_steps, n_length, x_val.shape[2]))

    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=5, activation='relu'), input_shape=(None, n_length, n_features)))
    # model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu')))
    model.add(TimeDistributed(Dropout(params['dropout'])))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(128, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))

    model.add(LSTM(128))  # returns a sequence of vectors of dimension 32
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.summary()
    # model.add(Conv2D(filters=2, kernel_size=(5, 2), data_format='channels_first',
    #                  activation='sigmoid',
    #                  padding='same'))
    # Lets train on 4 GPUs
    model = keras.utils.multi_gpu_model(model, gpus=[1, 2, 3, 4])

    model.compile(optimizer=get_optimizer(params['optimizer']),
                  loss='mean_absolute_error',
                  metrics=['mae'])

    model.summary()

    history = model.fit(x_train, y_train,
              validation_data=[x_val, y_val],
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=20, verbose=1, mode='min'),
                                   ModelCheckpoint(filepath='./Data/Save_models/CNN_LSTM/my_model_{}_{}_{}_{}.h5'.format(params['batch_size'], params['epochs'], params['dropout'], params['optimizer']),
                                                   monitor='val_loss', verbose=1, save_best_only=True, mode='min')])

    return history, model


if __name__ == '__main__':

    for variables in (['All_variables', 'Two_variables']):

        for prediction_horizon in ([20, 25, 30]):

            if variables == 'Two_variables':
                data_file = 'Data/modeling_data_{}_v1.h5'.format(prediction_horizon)

            if variables == 'All_variables':
                data_file = 'Data/modeling_data_{}.h5'.format(prediction_horizon)

            x_train, y_train, x_test, y_test = load_data_for_convLSTM(data_file)

            x_train = x_reshape(x_train)
            x_test = x_reshape(x_test)
            y_train = label_reshape(y_train)
            y_test = label_reshape(y_test)

            p = {'batch_size': [1000, 1200, 1400, 1600],
                 'epochs': [100, 150, 200, 250],
                 'dropout': (0.1, 0.5, 5),
                 'optimizer': ['Adam', 'Nadam']}

            h = ta.Scan(x=x_train, y=y_train, x_val=x_test, y_val=y_test,
                        model=corridor_cnn_lstm,
                        fraction_limit=.05,
                        params=p,
                        experiment_name='corridor_cnn_lstm')

            # accessing the results data frame
            h.data.head()
            h.data.to_csv('./Data/Save_models/CNN_LSTM/{}_server_{}_min/corridor_cnn_lstm.csv'.format(variables, prediction_horizon), sep=',')
            print('CNN_LSTM', variables, '{}_min'.format(prediction_horizon))

