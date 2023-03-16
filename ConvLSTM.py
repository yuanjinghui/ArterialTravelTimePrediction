"""
Created on Fri Jan 17 20:01:56 2020

@author: Jinghui Yuan: jinghuiyuan@knights.ucf.edu
"""
import h5py
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Dense, Dropout, CuDNNLSTM
from keras.callbacks import Callback, EarlyStopping
from keras.optimizers import Adam, Nadam, SGD, RMSprop, Adadelta
import talos as ta
from keras.callbacks import TensorBoard
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use id from $ nvidia-smi

def load_data_for_convLSTM():
    with h5py.File('Data/modeling_data_v1.h5', 'r') as f:
    # with h5py.File('Data/modeling_data.h5', 'r') as f:
        print(list(f.keys()))
        x_train = f['x_train'][:]
        y_train = f['y_train'][:]
        x_test = f['x_test'][:]
        y_test = f['y_test'][:]
        f.close()
    return x_train, y_train, x_test, y_test


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
def corridor_convlstm(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(ConvLSTM2D(filters=63, kernel_size=(5, 2), data_format='channels_first', activation='relu',
                         input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4]),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))

    model.add(ConvLSTM2D(filters=63, kernel_size=(5, 2), data_format='channels_first', activation='relu',
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))

    model.add(ConvLSTM2D(filters=126, kernel_size=(5, 2), data_format='channels_first', activation='relu',
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))

    model.add(ConvLSTM2D(filters=126, kernel_size=(5, 2), data_format='channels_first', activation='relu',
                         padding='same', return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))

    model.add(Conv2D(filters=1, kernel_size=(5, 2), data_format='channels_first',
                     activation='relu',
                     padding='same'))

    model.compile(loss='mean_squared_error', optimizer=get_optimizer(params['optimizer']), metrics=['mae'])

    history = model.fit(x_train, y_train,
              validation_data=[x_val, y_val],
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=10, verbose=1, mode='min'),
                                   ModelCheckpoint(filepath='./Data/Save_models/my_model_{}_{}_{}_{}.h5'.format(params['batch_size'], params['epochs'], params['dropout'], params['optimizer']),
                                                   monitor='val_loss', verbose=1, save_best_only=True, mode='min')])

    # history = model.fit(x_train, y_train,
    #                     validation_data=[x_val, y_val],
    #                     batch_size=params['batch_size'],
    #                     epochs=params['epochs'],
    #                     callbacks=[roc_callback(training_data=(x_train, y_train), validation_data=(x_val, y_val)),
    #                                tensorboard, early_stopper(params['epochs'], monitor='roc_auc', mode=[0.001, 5]),
    #                                ModelCheckpoint(filepath='./Data/Save_models/Group1_intersection/my_model_{}_{}_{}.h5'.format(params['batch_size'], params['epochs'], params['dropout']),
    #                                                monitor='roc_auc', verbose=1, save_best_only=True, mode='max')])

    # history = model.fit(x_train, y_train,
    #           # validation_data=[x_test, y_test],
    #           batch_size=params['batch_size'],
    #           epochs=params['epochs'],
    #           callbacks=[roc_callback(training_data=(x_train, y_train), validation_data=(x_val, y_val)),
    #                      early_stopper(epochs=params['epochs'], mode=[0.001, 5], monitor='roc_auc'),
    #                                ModelCheckpoint(filepath='./Data/Save_models/my_model_{}_{}_{}_{}.h5'.format(params['lr'], params['batch_size'], params['epochs'], params['dropout']),
    #                                                monitor='roc_auc', verbose=1, save_best_only=True, mode='max')])

    return history, model


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = load_data_for_convLSTM()

    p = {'batch_size': [1000, 1200, 1400, 1600],
         'epochs': [100, 150, 200, 250],
         'dropout': (0.1, 0.5, 10),
         'optimizer': ['Adam', 'Nadam', 'RMSprop', 'Adadelta']}

    h = ta.Scan(x=x_train, y=y_train, x_val=x_test, y_val=y_test,
                model=corridor_convlstm,
                grid_downsample=.3,
                params=p,
                dataset_name='corridor_convlstm',
                experiment_no='1')

    # accessing the results data frame
    h.data.head()
    h.data.to_csv('./Data/Save_models/corridor_convlstm.csv', sep=',')


