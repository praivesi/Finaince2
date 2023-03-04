import tensorflow as tf

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import datetime

from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping

normalizer = preprocessing.MinMaxScaler()

start_date = datetime.datetime.now() - datetime.timedelta(days=365*20)

df_heelim = fdr.DataReader('037440', start_date)

prep_heelim = df_heelim
prep_heelim['Volume'] = df_heelim['Volume'].replace(0, np.nan)
prep_heelim = prep_heelim.dropna()
prep_heelim

# plt.figure(figsize=(7,4))

# plt.title('heelim')
# plt.ylabel('price (won)')
# plt.xlabel('period (day)')
# plt.grid()

# plt.plot(df_heelim['Close'], label='Close', color='r')
# plt.legend(loc='best')

# plt.show()

norm_cols = ['Open', 'Close', 'Volume']
norm_heelim_np = normalizer.fit_transform(prep_heelim[norm_cols])

norm_heelim = pd.DataFrame(norm_heelim_np, columns=norm_cols)
norm_heelim

# plt.plot(norm_heelim['Close'], label='Close', color='purple')
# plt.show()

feature_cols = ['Open', 'Close', 'Volume']
label_cols = ['Close']

feature_df = pd.DataFrame(norm_heelim, columns=feature_cols)
label_df = pd.DataFrame(norm_heelim, columns=label_cols)

feature_np = feature_df.to_numpy()
label_np = label_df.to_numpy()

def make_sequence_dataset(feature, label, window_size):
    feature_list = []
    label_list = []

    for i in range(len(feature) - window_size):
        feature_list.append(feature[i:i + window_size])
        label_list.append(label[i + window_size])
    
    return np.array(feature_list), np.array(label_list)

window_size = 40

X, Y = make_sequence_dataset(feature_np, label_np, window_size)

split = -200

x_train = X[:split]
y_train = Y[:split]

x_test = X[split:]
y_test = Y[split:]

input_size = window_size
sequence_length = 3 # Open, Close, Volume
num_layers = 2
hidden_size = 256
num_classes = 1
learning_rate = 0.001
batch_size = 64
num_epochs = 2



model = Sequential()
model.add(LSTM(128, activation='tanh', input_shape=x_train[0].shape))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=16, callbacks=[early_stop])

pred = model.predict(x_test)

plt.figure(figsize=(12,6))
plt.plot(y_test, label='actual')
plt.plot(pred, label='prediction')
plt.grid()
plt.legend(loc='best')

plt.show()