from os import execlpe
from threading import active_count
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# CNN-LSTM model for univariate time series forecasting


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_x = i + n_steps
        if end_x > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_x], sequence[end_x]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_steps = 4
X, y = split_sequence(seq, n_steps)
n_features = 1
n_seq = 2
n_steps = 2
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1,
          activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=500, verbose=0)

x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
